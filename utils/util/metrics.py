import numpy as np
import torch
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_fscore_support,
)
from utils.util.logger import setup_logger
import os

def get_metrics(logits,labels,y_true=None, y_scores=None,test_info = None,save_folder=None,overwrite_log = False):
    # save_folder 是验证效果保存的文件夹
    macro_metrics = subject_accuracy(logits, labels)
    topk_metrics = {
            "top1": topk_acc(logits, labels, k=1).item(),
            "top3": topk_acc(logits, labels, k=3).item(),
            "top5": topk_acc(logits, labels, k=5).item(),
        }
    # 1:1验证
    if y_true is None or y_scores is None:
        eval_metrics = eval_allpairs_eer(sim_matrix=logits, labels=labels, save_folder=save_folder)
    else:
        eval_metrics = compute_eer(y_true=y_true, y_score=y_scores, save_folder=save_folder)
        eval_metrics['verify_eer'] = eval_metrics['eer']
        del eval_metrics['eer']
        eval_metrics['verify_auc'] = eval_metrics['auc']
        del eval_metrics['auc']
        del eval_metrics['threshold']
    all_metrics = {**eval_metrics, **macro_metrics,**topk_metrics}
    keys_to_extract = ['top1', 'top3', 'top5','precision',
                       'recall','f1', 'verify_eer','verify_auc']
    all_metrics = {k: v for k, v in all_metrics.items() if k in keys_to_extract}
    value_list = []
    header_list = []
    for k in keys_to_extract:
        v = all_metrics.get(k)
        header_list.append(k)
        if isinstance(v, (float, int)):
            value_list.append(f"{v:.4f}")
        
    print(",".join(header_list))
    print(",".join(value_list))
    logger = setup_logger(log_dir=f"metrics/{test_info}", log_filename=f"metrics.log",overwrite_log = overwrite_log)
    # logger.info(",".join(header_list))
    logger.info(",".join(value_list))

def topk_acc(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 1, **kwargs
) -> torch.Tensor:
    num_classes = logits.shape[1]
    k = min(k, num_classes)
    _, top_k_indices = torch.topk(logits, k=k, dim=1, largest=True, sorted=False)
    correct = top_k_indices.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean()

def per_subject_eer(similarity_matrix, test_labels):
    similarity_matrix = np.array(similarity_matrix)
    test_labels = np.array(test_labels)

    n_samples, n_templates = similarity_matrix.shape
    unique_subjects = np.unique(test_labels)
    results = []

    for subj_id in unique_subjects:
        if subj_id >= n_templates or subj_id < 0:
            print(
                f"Skipping Subject ID {subj_id}: Out of template range (0-{n_templates-1})"
            )
            continue
        row_indices = np.where(test_labels == subj_id)[0]

        genuine_scores = similarity_matrix[row_indices, subj_id]

        col_mask = np.ones(n_templates, dtype=bool)
        col_mask[subj_id] = False

        imposter_scores = similarity_matrix[row_indices][:, col_mask].flatten()

        if len(genuine_scores) == 0 or len(imposter_scores) == 0:
            continue

        y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])
        y_score = np.concatenate([genuine_scores, imposter_scores])
        data_dict = compute_eer(y_true, y_score)

        results.append(
            {
                "Subject_ID": subj_id,
                "per_EER": data_dict["eer"],
                "per_Threshold": data_dict["threshold"],
                "per_auc": data_dict["auc"],
                "per_Genuine_Mean": np.mean(genuine_scores),
                "per_Imposter_Mean": np.mean(imposter_scores),
            }
        )

    return results


def subject_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    # 转换为 numpy 数组
    preds = torch.argmax(logits, dim=1)
    labels_np = labels.numpy()

    # 检查空数据集
    if len(labels_np) == 0:
        return {
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "class_accuracy": {},
        }

    present_classes = np.unique(labels_np)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds, average='macro', zero_division=0
    )

    _, recalls, _, _ = precision_recall_fscore_support(
        labels_np, preds, labels=present_classes, average=None, zero_division=0
    )
    class_accuracy = []
    for i, subj_id in enumerate(present_classes):
        class_accuracy.append({
            "Subject_ID": int(subj_id),
            "acc": float(recalls[i]), # 使用枚举索引 i 而非标签值 subj_id
        })

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "class_accuracy": class_accuracy,
    }

    return metrics

def compute_eer(y_true, y_score, save_folder=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    auc_value = auc(fpr, tpr)
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    best_thr = thresholds[idx]
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        np.save(f"{save_folder}/fpr.npy", fpr)
        np.save(f"{save_folder}/tpr.npy", tpr)
        np.save(f"{save_folder}/thresholds.npy", thresholds)
        plot_roc(fpr, tpr, save_folder)
        plot_far_frr(fpr, tpr, thresholds, save_folder)
    return {'eer': eer, 'threshold': best_thr, 'auc': auc_value}    

def eval_allpairs_eer(sim_matrix, labels,save_folder=None):
    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.detach().cpu().numpy()
    elif isinstance(sim_matrix, list):
        if len(sim_matrix) > 0 and isinstance(sim_matrix[0], torch.Tensor):
            sim_matrix = torch.stack(sim_matrix).detach().cpu().numpy()
        else:
            sim_matrix = np.array(sim_matrix)

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    sim_matrix = np.asarray(sim_matrix)
    labels = np.asarray(labels).astype(int)

    N, C = sim_matrix.shape
    assert labels.shape[0] == N, f"labels 长度({labels.shape[0]})与 N({N})不一致"
    assert np.all((labels >= 0) & (labels < C)), "labels 超出类别范围"

    label_mask = np.zeros((N, C), dtype=bool)
    label_mask[np.arange(N), labels] = True

    genuine_scores = sim_matrix[label_mask]      # shape: (N,)
    imposter_scores = sim_matrix[~label_mask]    # shape: (N*(C-1),)

    y_scores = np.concatenate([genuine_scores, imposter_scores])
    y_true = np.concatenate([
        np.ones_like(genuine_scores, dtype=int),
        np.zeros_like(imposter_scores, dtype=int)
    ])
    eer_dict = compute_eer(y_true, y_scores,save_folder)

    print("-" * 30)
    print(f"All-pairs Verification")
    print(f"#Genuine: {genuine_scores.shape[0]}  #Imposter: {imposter_scores.shape[0]}")
    print(f"EER: {eer_dict['eer'] * 100:.2f}%")
    print(f"Threshold: {eer_dict['threshold']:.4f}  AUC: {eer_dict['auc']:.4f}")

    return {
        "verify_eer": eer_dict["eer"],
        "verify_auc": eer_dict["auc"],
        "threshold": eer_dict["threshold"],
    }