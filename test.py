import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
from collections import OrderedDict

# ================= 导入部分 =================
from model_v11 import CAS_UNet_V11
from dataset import PathologyDataset, get_transforms

# ================= 配置区域 =================
# V11 最佳权重路径
CHECKPOINT_PATH = "/IMBR_Data/Student-home/2024M_WangHaiyu/SLED-main/Newway2016/14-1/V14/best_model.pth"

# 测试集路径 (保持不变)
BASE_TEST = Path("/IMBR_Data/Student-home/2024M_WangHaiyu/SLED-main/ISIC2016/test")

IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ===========================================

def calculate_metrics(pred, target, epsilon=1e-6):
    """计算六大指标"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    TP = (pred * target).sum()
    TN = ((1 - pred) * (1 - target)).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()
    
    dice = (2. * TP + epsilon) / (2. * TP + FP + FN + epsilon)
    iou = (TP + epsilon) / (TP + FP + FN + epsilon)
    acc = (TP + TN + epsilon) / (TP + TN + FP + FN + epsilon)
    pre = (TP + epsilon) / (TP + FP + epsilon)
    rec = (TP + epsilon) / (TP + FN + epsilon)
    spe = (TN + epsilon) / (TN + FP + epsilon)
    
    return {
        'Dice': dice.item(), 'IoU': iou.item(), 'Acc': acc.item(),
        'Pre': pre.item(), 'Rec': rec.item(), 'Spe': spe.item()
    }

@torch.no_grad()
def tta_inference(model, image):
    """
    TTA 策略：原图 + H翻转 + V翻转 -> 平均
    """
    model.eval()
    
    # 1. 原图
    pred_1 = torch.sigmoid(model(image))
    
    # 2. 水平翻转
    img_h = torch.flip(image, [3])
    pred_h = torch.sigmoid(model(img_h))
    pred_2 = torch.flip(pred_h, [3])
    
    # 3. 垂直翻转
    img_v = torch.flip(image, [2])
    pred_v = torch.sigmoid(model(img_v))
    pred_3 = torch.flip(pred_v, [2])
    
    # 平均结果
    return (pred_1 + pred_2 + pred_3) / 3.0

def run_evaluation():
    print("=" * 60)
    print(f"正在评估 CAS-UNet V11 (DFG-Fusion) + TTA")
    print(f"权重: {CHECKPOINT_PATH}")
    print("=" * 60)

    # 1. 初始化模型
    model = CAS_UNet_V11(in_channels=3, num_classes=1, base_ch=32)
    
    # 2. 加载权重
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: 找不到权重文件 {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # 兼容性处理
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 处理 DataParallel 前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') 
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    # 3. 数据集
    dataset = PathologyDataset(
        BASE_TEST / 'images', BASE_TEST / 'masks',
        transform=get_transforms(IMG_SIZE, is_train=False),
        img_size=IMG_SIZE
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. 测试循环
    metrics_history = {'Dice': [], 'IoU': [], 'Acc': [], 'Pre': [], 'Rec': [], 'Spe': []}
    
    with torch.no_grad():
        for batch in tqdm(loader):
            img = batch['image'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            # TTA 推理
            pred_prob = tta_inference(model, img)
            
            # 二值化
            pred_bin = (pred_prob > 0.5).float()
            
            batch_metrics = calculate_metrics(pred_bin, mask)
            for k, v in batch_metrics.items():
                metrics_history[k].append(v)
            
    # 5. 结果输出
    print("\n" + "="*60)
    print(f"{'Metric':<10} | {'Mean':<10} | {'Std':<10}")
    print("-" * 60)
    
    results = {}
    for k, v in metrics_history.items():
        mean_val = np.mean(v)
        std_val = np.std(v)
        results[k] = mean_val
        print(f"{k:<10} | {mean_val:.4f}     | {std_val:.4f}")
        
    print("="*60)
    print("\n[V11 Latex Format]:")
    print(f"{results['Dice']:.4f} & {results['IoU']:.4f} & {results['Acc']:.4f} & {results['Pre']:.4f} & {results['Rec']:.4f} & {results['Spe']:.4f}")

if __name__ == '__main__':
    run_evaluation()