import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PathologyDataset(Dataset):
    """
    病理图像数据集 - 终极适配版
    能够自动处理 jpg/png 后缀不一致问题，以及 masks 子目录问题
    """
    def __init__(self, image_dir, mask_dir, artifact_dirs=None, transform=None, img_size=512):
        # 支持 jpg 和 png
        self.image_paths = sorted(list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png')))
        self.mask_dir = Path(mask_dir)
        self.artifact_dirs = artifact_dirs
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def extract_edge(self, mask):
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        edge = mask - eroded
        return edge
    
    def _load_mask_robust(self, base_dir, stem):
        """
        穷举法查找 Mask 文件
        """
        # 1. 检查 base_dir/masks/stem.jpg (最常见伪影格式)
        p = base_dir / "masks" / f"{stem}.jpg"
        if p.exists(): return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        
        # 2. 检查 base_dir/masks/stem.png (常见GT格式)
        p = base_dir / "masks" / f"{stem}.png"
        if p.exists(): return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        
        # 3. 检查 base_dir/stem.jpg
        p = base_dir / f"{stem}.jpg"
        if p.exists(): return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        
        # 4. 检查 base_dir/stem.png
        p = base_dir / f"{stem}.png"
        if p.exists(): return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        
        # 5. 检查 base_dir/stem_segmentation.png (ISIC GT 特有)
        p = base_dir / f"{stem}_Segmentation.png"
        if p.exists(): return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

        return None

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = img_path.stem
        
        # 1. 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 2. 读取 GT Mask
        mask = self._load_mask_robust(self.mask_dir, stem)
        if mask is not None:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8)
        else:
            # 极少数情况找不到GT，给全黑，并在控制台忽略（或报错）
            mask = np.zeros((h, w), dtype=np.uint8)
            
        # 3. 生成 ROI Mask (合并所有伪影)
        # 默认全为1 (有效)
        roi_mask = np.ones((h, w), dtype=np.uint8)
        
        if self.artifact_dirs:
            for art_cfg in self.artifact_dirs:
                art_dir = Path(art_cfg['dir'])
                art = self._load_mask_robust(art_dir, stem)
                
                if art is not None:
                    art = cv2.resize(art, (w, h), interpolation=cv2.INTER_NEAREST)
                    # 逻辑: 伪影文件中，白色(255)是有效区域，黑色(0)是伪影
                    # 这一点非常关键，基于你之前的 dark_corner 代码逻辑
                    valid_area = (art > 127).astype(np.uint8)
                    roi_mask = cv2.bitwise_and(roi_mask, valid_area)

        # 4. 提取边缘
        edge = self.extract_edge(mask)
        
        # 5. 数据增强 (必须同步变换 ROI)
        if self.transform:
            # 修复：使用正确的 additional_targets 方式
            augmented = self.transform(
                image=image, 
                mask=mask,
                edge=edge,
                roi=roi_mask
            )
            image = augmented['image']
            mask_out = augmented['mask']
            edge_out = augmented['edge']
            roi_out = augmented['roi']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_out = torch.from_numpy(mask).float()
            edge_out = torch.from_numpy(edge).float()
            roi_out = torch.from_numpy(roi_mask).float()
            
        # 确保是 tensor 类型
        if isinstance(mask_out, np.ndarray):
            mask_out = torch.from_numpy(mask_out).float()
        if isinstance(edge_out, np.ndarray):
            edge_out = torch.from_numpy(edge_out).float()
        if isinstance(roi_out, np.ndarray):
            roi_out = torch.from_numpy(roi_out).float()
            
        # 增加维度 [H, W] -> [1, H, W]
        if mask_out.ndim == 2: mask_out = mask_out.unsqueeze(0)
        if edge_out.ndim == 2: edge_out = edge_out.unsqueeze(0)
        if roi_out.ndim == 2: roi_out = roi_out.unsqueeze(0)
            
        return {
            'image': image,
            'mask': mask_out.float(),
            'edge': edge_out.float(),
            'roi': roi_out.float()
        }

def get_transforms(img_size=512, is_train=True):
    # 修复：正确配置 additional_targets
    additional_targets = {
        'edge': 'mask',
        'roi': 'mask'
    }
    
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets=additional_targets)
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets=additional_targets)
