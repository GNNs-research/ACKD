import numpy as np
import copy
import torch
import dgl
import time
import torch.nn.functional as F
from utils import set_seed
from scipy.stats import entropy
# adv_deltas=FGSM(model, feats[idx_batch[i]], labels[idx_batch[i]], criterion)
def FGSM(model, inputs, targets, criterion, conf):
    """
    Fast Gradient Sign Method (FGSM) for generating adversarial perturbations.

    Parameters:
    - model: 模型
    - inputs: 输入数据
    - targets: 输入数据的真实标签
    - criterion: 损失函数
    - epsilon: 扰动的幅度

    Returns:
    - adv_deltas: 生成的对抗性扰动
    """
    inputs.requires_grad = True

    _,logits= model(None,inputs)
    outputs = logits.log_softmax(dim=1)
    loss = criterion(outputs, targets)

    model.zero_grad()
    loss.backward()
    data_grad = inputs.grad.data
    epsilon = conf["adv_eps"]
    adv_deltas = epsilon * data_grad.sign()

    return adv_deltas


def train_mini_batch_stu(model, feats, labels,teacher_emb, batch_size, criterion, optimizer,conf,args, lamb=1):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    device = feats.device
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        # No graph needed for the forward function
        batch_mlp_emb,logits = model(None, feats[idx_batch[i]])
        out = logits.log_softmax(dim=1)

        if conf["contrative_distillation"]:
            batch_mlp_emb = batch_mlp_emb[-1]

            batch_mlp_emb = model.encode_mlp4kd(batch_mlp_emb)

            batch_teacher_emb = teacher_emb[idx_batch[i]]
            batch_teacher_emb = model.encode_teacher4kd(batch_teacher_emb)

            batch_mlp_emb = F.normalize(batch_mlp_emb, p=2, dim=-1)
            batch_teacher_emb = F.normalize(batch_teacher_emb, p=2, dim=-1)

            nce_logits = torch.mm(batch_mlp_emb, batch_teacher_emb.transpose(0, 1))#(140,140)的相似度矩阵
            nce_labels = torch.arange(batch_mlp_emb.shape[0])#140个数字(140,)

            loss_nce = F.cross_entropy(nce_logits / 0.075, nce_labels.to(device))

        if conf["FGSM_adv"]:
            adv_deltas=FGSM(model, feats[idx_batch[i]], labels[idx_batch[i]], criterion,conf)
            adv_feats = torch.add(feats[idx_batch[i]], adv_deltas)
            _,adv_logits = model(None, adv_feats)

            adv_out = adv_logits.log_softmax(dim=1)
            loss_adv = criterion(adv_out, labels[idx_batch[i]])


        loss_label = criterion(out, labels[idx_batch[i]])
        loss = loss_label
        if conf["contrative_distillation"]:
            loss +=  conf["contrative_distillation_weight"]*loss_nce


        if conf["FGSM_adv"]:
            loss += conf["adv_distillation_weight"] * loss_adv


        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


def evaluate_mini_batch_stu(
    model, feats, labels, criterion, batch_size, evaluator, idx_eval=None
):
    """
    Evaluate MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    iterations = 1  # 重复计算的轮次
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    model.eval()
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            num_batches = int(np.ceil(len(feats) / batch_size))
            out_list = []

            for i in range(num_batches):
                _,logits = model.inference(None, feats[batch_size * i : batch_size * (i + 1)])
                out = logits.log_softmax(dim=1)
                out_list += [out.detach()]
            out_all = torch.cat(out_list)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            if idx_eval is None:
                loss = criterion(out_all, labels)
                score = evaluator(out_all, labels)
            else:
                loss = criterion(out_all[idx_eval], labels[idx_eval])
                score = evaluator(out_all[idx_eval], labels[idx_eval])


    return out_all, loss.item(), score, curr_time




"""
3. Distill
"""


def distill_run_transductive(
    conf,
    model,
    feats,
    labels,
    out_t_all,
    teacher_emb,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
    args
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    idx_l, idx_t, idx_val, idx_test = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    teacher_emb = teacher_emb.to(device)

    feats_l, labels_l = feats[idx_l], labels[idx_l]
    feats_t, out_t = feats[idx_t], out_t_all[idx_t]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]
    teacher_emb_l,teacher_emb_t = teacher_emb[idx_l],teacher_emb[idx_t]

    best_epoch, best_score_val, count = 0, 0, 0

    for epoch in range(1, conf["max_epoch"] + 1):

        loss_l = train_mini_batch_stu(model, feats_l, labels_l,teacher_emb_l, batch_size, criterion_l, optimizer, conf,args,lamb)
        loss_t = train_mini_batch_stu(model, feats_t, out_t,teacher_emb_t, batch_size, criterion_t, optimizer,conf, args,1 - lamb)
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l,_ = evaluate_mini_batch_stu(model, feats_l, labels_l, criterion_l, batch_size, evaluator)
            _, loss_val, score_val,_ = evaluate_mini_batch_stu(model, feats_val, labels_val, criterion_l, batch_size, evaluator)
            _, loss_test, score_test,_ = evaluate_mini_batch_stu(model, feats_test, labels_test, criterion_l, batch_size, evaluator)

            logger.info(f"Ep {epoch:3d} | loss: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}")
            loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    out, _, score_val , inference_time= evaluate_mini_batch_stu(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_val
    )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels_test)

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )

    return out, score_val, score_test,inference_time


def distill_run_inductive(
    conf,
    model,
    feats,
    labels,
    out_t_all,
    teacher_emb,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
    args,
):
    """
    Distill training and eval under the inductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively.
    loss_and_score: Stores losses and scores.
    通过 distill_indices 指定了硬标签训练、软标签训练、验证和测试集的划分。
    索引以 obs_idx_ 开头的包含了观察图 obs_g 中的节点索引。
    索引以 idx_ 开头的包含了原始图 g 中的节点索引。
    模型在观察图 obs_g 上进行训练，并在观察测试节点 (obs_idx_test) 和归纳测试节点 (idx_test_ind) 上进行评估。
    假设输入图很大，并且 MLP 是学生模型。因此，仅使用节点特征和小批量训练。
    idx_obs：原始图 g 中的节点索引，构成观察图 obs_g。
    out_t：教师模型生成的软标签。
    criterion_l 和 criterion_t：分别用于硬标签 (labels) 和软标签 (out_t) 的损失函数。
    loss_and_score：存储损失和评分。
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    teacher_emb = teacher_emb.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]

    #       obs_idx_l = obs_idx_train
    #         obs_idx_t = torch.cat([obs_idx_train, obs_idx_val, obs_idx_test])
    #         distill_indices = (
    #             obs_idx_l,
    #             obs_idx_t,
    #             obs_idx_val,
    #             obs_idx_test,
    #             idx_obs,
    #             idx_test_ind,
    #         )
    feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, out_t = obs_feats[obs_idx_t], obs_out_t[obs_idx_t]
    teacher_emb_l,teacher_emb_t=teacher_emb[obs_idx_l],teacher_emb[obs_idx_t]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train_mini_batch_stu(model, feats_l, labels_l,teacher_emb_l, batch_size, criterion_l, optimizer, conf,args,lamb)
        loss_t = train_mini_batch_stu(model, feats_t, out_t, teacher_emb_t,batch_size, criterion_t, optimizer,conf, args,1 - lamb)
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l,_ = evaluate_mini_batch_stu(model, feats_l, labels_l, criterion_l, batch_size, evaluator)
            _, loss_val, score_val,_ = evaluate_mini_batch_stu(model, feats_val, labels_val, criterion_l, batch_size, evaluator)
            _, loss_test_tran, score_test_tran,_ = evaluate_mini_batch_stu(
                model,
                feats_test_tran,
                labels_test_tran,
                criterion_l,
                batch_size,
                evaluator,
            )
            _, loss_test_ind, score_test_ind,_ = evaluate_mini_batch_stu(
                model,
                feats_test_ind,
                labels_test_ind,
                criterion_l,
                batch_size,
                evaluator,
            )

            logger.debug(
                f"Ep {epoch:3d} | l: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_l,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_l,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    obs_out, _, score_val,_ = evaluate_mini_batch_stu(
        model, obs_feats, obs_labels, criterion_l, batch_size, evaluator, obs_idx_val
    )
    out, _, score_test_ind,inference_time = evaluate_mini_batch_stu(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_test_ind
    )

    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test_tran = evaluator(obs_out[obs_idx_test], labels_test_tran)
    out[idx_obs] = obs_out

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d} score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )

    return out, score_val, score_test_tran, score_test_ind,inference_time
