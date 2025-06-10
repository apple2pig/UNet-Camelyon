# 计算单张图片的指标
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import segmentation_models_pytorch as smp
IoU = smp.utils.metrics.IoU()
Acc = smp.utils.metrics.Accuracy()

def calculate_Mission_indicators(mask_in, predicted_in):
    acc_list = []
    dice_list = []
    iou_list = []
    auc_list = []


    batch_size, _, _, _ = mask_in.shape
    for index in range(0, batch_size):
        # 一个向量展平成一维
        mask = mask_in[index].reshape(-1)
        predicted = predicted_in[index].reshape(-1)

        TP = torch.eq(mask[torch.eq(mask, predicted)], 1).sum()
        FN = torch.eq(mask[torch.ne(mask, predicted)], 1).sum()
        FP = torch.eq(mask[torch.ne(mask, predicted)], 0).sum()
        TN = torch.eq(mask[torch.eq(mask, predicted)], 0).sum()

        # acc_1 = (TP + TN) / (TP + FN + FP + TN)
        dice_1 = (2 * TP) / (2 * TP + FP + FN)
        # iou_1 = TP / (TP + FP + FN)
        iou_1 = IoU(mask, predicted)
        acc_1 = Acc(mask, predicted)
        
        acc_list.append(acc_1.item())
        dice_list.append(dice_1.item())
        iou_list.append(iou_1.item())
        # auc_list.append(roc_auc_score(mask.cpu().numpy(), predicted.cpu().numpy()))

    return acc_list, dice_list, iou_list, auc_list

# 批次处理
def calculate_acc(model, val_dataloaders, device):
    acc_list = []
    dice_list = []
    iou_list = []
    auc_list = []

    model.eval()
    with tqdm(total=len(val_dataloaders), desc='Val', colour='blue') as pbar:
        with torch.no_grad():
            for x, y in val_dataloaders:
                inputs = x.to(device)
                labels = y.to(device)
                
                predicted = model(inputs)
                predicted = torch.where(predicted > 0.5, torch.ones_like(predicted), torch.zeros_like(predicted))  # 二值化
                mask = torch.where(labels > 0.5, torch.ones_like(labels), torch.zeros_like(labels))
                
                temp_acc_list, temp_dice_list, temp_iou_list, temp_auc_list = calculate_Mission_indicators(
                    mask.float(), predicted.float())
                
                acc_list.extend(temp_acc_list)
                iou_list.extend(temp_iou_list)
                dice_list.extend(temp_dice_list)
                auc_list.extend(temp_auc_list)
                
                acc = np.mean(acc_list)
                dice = np.mean(dice_list)
                iou = np.mean(iou_list)
                auc = np.mean(auc_list)

                pbar.set_postfix(**{"AUC": f"{auc:.4f}",
                                "Acc": f"{acc:.4f}",
                                "Dice": f"{dice:.4f}",
                                "Iou": f"{iou:.4f}"})
                pbar.update(1)

        pbar.close()

        return acc, dice, iou, auc
