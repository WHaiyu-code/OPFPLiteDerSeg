import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import argparse
import logging
from pathlib import Path

# 导入自定义模块 (注意这里导入的是 V11)
from model_v11 import CAS_UNet_V11
from loss_v10 import CombinedLossV10 # Loss 依然用 V10 的
from dataset import PathologyDataset, get_transforms 

# ---------------- 配置区域 ----------------
TRAIN_IMG_DIR = '/IMBR_Data/Student-home/2024M_WangHaiyu/SLED-main/ISIC2016/train/images'
TRAIN_MASK_DIR = '/IMBR_Data/Student-home/2024M_WangHaiyu/SLED-main/ISIC2016/train/masks'
ARTIFACT_DIRS = None # 保持 None，让 CBAM 和 DFG 自动学习去噪

TEST_IMG_DIR = '/IMBR_Data/Student-home/2024M_WangHaiyu/SLED-main/ISIC2016/test/images'
TEST_MASK_DIR = '/IMBR_Data/Student-home/2024M_WangHaiyu/SLED-main/ISIC2016/test/masks'

# [重点] 专门的保存路径，用于存放 V11 的训练统计和权重
SAVE_DIR = '/IMBR_Data/Student-home/2024M_WangHaiyu/SLED-main/Newway2016/14-1/V14' 

def calculate_dice(pred_logit, mask):
    pred = torch.sigmoid(pred_logit) > 0.5
    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum()
    return (2. * inter + 1e-5) / (union + 1e-5)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    dice_scores = []
    
    for batch in tqdm(loader, desc="Validating"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward
        output = model(images)
        
        for i in range(output.shape[0]):
            d = calculate_dice(output[i], masks[i])
            dice_scores.append(d.item())
            
    return np.mean(dice_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4) 
    parser.add_argument('--img_size', type=int, default=256)
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 配置日志文件 (保留节点信息)
    logging.basicConfig(
        filename=os.path.join(SAVE_DIR, "log.txt"), 
        level=logging.INFO,
        format='[%(asctime)s] %(message)s', 
        datefmt='%H:%M:%S'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Saving checkpoints to: {SAVE_DIR}")
    
    # 1. Dataset
    train_dataset = PathologyDataset(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, 
        artifact_dirs=ARTIFACT_DIRS,
        transform=get_transforms(args.img_size, is_train=True),
        img_size=args.img_size
    )
    val_dataset = PathologyDataset(
        TEST_IMG_DIR, TEST_MASK_DIR, 
        transform=get_transforms(args.img_size, is_train=False),
        img_size=args.img_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}")
    
    # 2. Model (V11 with DFG)
    print("Initializing CAS-UNet V11 (DFG-Fusion + CBAM)...")
    model = CAS_UNet_V11(in_channels=3, num_classes=1, base_ch=32).to(device)
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = CombinedLossV10().to(device)
    
    best_dice = 0.0
    
    # 4. Loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        dice = validate(model, val_loader, device)
        
        scheduler.step()
        
        # 打印并保存日志
        log_msg = f"Epoch {epoch} | Loss: {loss:.4f} | Val Dice: {dice:.4f} | Best: {best_dice:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
        print(log_msg)
        logging.info(log_msg)
        
        # 保存最佳节点
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print(">>> New Best Model Saved!")
            logging.info(">>> New Best Model Saved!")
            
        # 保存最新节点 (用于断点续训)
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'last_model.pth'))

if __name__ == '__main__':
    main()