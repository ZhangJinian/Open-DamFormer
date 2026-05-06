import torch

def compute_confusion_matrix(pred, target, num_classes, ignore_index=None):
    with torch.no_grad():
        mask = target >= 0
        if ignore_index is not None:
            mask &= (target != ignore_index)

        pred = pred[mask]
        target = target[mask]

        conf_mat = torch.bincount(
            num_classes * target + pred,
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)

    # print( conf_mat)
    return conf_mat


def compute_iou_from_confmat(conf_mat):
    """
    conf_mat: [C,C]
    """
    eps = 1e-6
    TP = torch.diag(conf_mat)
    FP = conf_mat.sum(0) - TP
    FN = conf_mat.sum(1) - TP

    iou = TP / (TP + FP + FN + eps)
    return iou


def compute_accuracy_from_confmat(conf_mat):
    correct = torch.diag(conf_mat).sum().float()
    total = conf_mat.sum().float()
    return correct / (total + 1e-6)

def compute_metrics_from_confmat(conf_mat, eps=1e-6):
    """
    conf_mat: [C, C] torch.Tensor (long or float)
    """
    conf_mat = conf_mat.float()
    C = conf_mat.size(0)

    TP = torch.diag(conf_mat)
    FP = conf_mat.sum(dim=0) - TP
    FN = conf_mat.sum(dim=1) - TP
    TN = conf_mat.sum() - (TP + FP + FN)

    # ---- basic metrics ----
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    iou = TP / (TP + FP + FN + eps)

    # ---- global metrics ----
    OA = TP.sum() / (conf_mat.sum() + eps)
    mIoU = iou.mean()

    freq = conf_mat.sum(dim=1) / (conf_mat.sum() + eps)
    FWIoU = (freq * iou).sum()

    metrics = {
        "OA": OA.item(),
        "mIoU": mIoU.item(),
        "FWIoU": FWIoU.item(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy(),
        "iou": iou.cpu().numpy(),
    }

    return metrics