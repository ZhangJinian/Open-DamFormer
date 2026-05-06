"""DamFormer 验证集推理脚本 - 使用固定验证集BRIGHT_val"""
import os
import sys
sys.path.insert(0, '/media/lzd/0A5CED4894A8F8FB/zjn/DisasterDetection/DamFormer')
sys.path.insert(0, '/media/lzd/0A5CED4894A8F8FB/zjn/DisasterDetection/FusionDamNet')

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

from damformer import DamFormer,MyDamFormer
from dataset import FusionDamDataset

# 指定使用GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 颜色映射
COLOR_MAP = {
    0: [255, 255, 255],  # No damage - white
    1: [70, 181, 121],   # Intact - green
    2: [228, 189, 139],  # Damaged - orange
    3: [182, 70, 69]     # Destroyed - red
}

def colorize_prediction(pred_mask):
    """将预测mask转换为彩色图像"""
    h, w = pred_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in COLOR_MAP.items():
        mask = (pred_mask == class_id)
        color_img[mask] = color
    
    return color_img

def main():
    print("="*70)
    print("DamFormer 验证集推理 (使用固定验证集)")
    print("="*70)
    
    # 路径配置 - 使用固定的验证集文件夹
    VAL_PATH = '/media/lzd/0A5CED4894A8F8FB/zjn/DisasterDetection/BRIGHT_val'
    CKPT_PATH = '/media/lzd/0A5CED4894A8F8FB/zjn/DisasterDetection/DamFormer/results/MyDamformer/ckpt/best.pth'
    # /media/lzd/0A5CED4894A8F8FB/zjn/DisasterDetection/DamFormer/results/MyDamformer/ckpt/best.pth
    OUTPUT_DIR = '/media/lzd/0A5CED4894A8F8FB/zjn/DisasterDetection/DamFormer/results/inference_val'
    RAW_DIR = os.path.join(OUTPUT_DIR, 'raw')
    COLOR_DIR = os.path.join(OUTPUT_DIR, 'color')
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(COLOR_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载固定的验证集 - 不使用splitfile，直接读取BRIGHT_val文件夹
    print("✓ Loading fixed validation set...")
    val_dataset = FusionDamDataset(VAL_PATH, split='train', use_splitfile=False, img_size=1024)
    
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Dataset path: {VAL_PATH}")
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # 加载模型
    print("✓ Loading DamFormer model...")
    # model = DamFormer(in_channels=3, num_classes=4).to(device)
    model = MyDamFormer(in_channels=3, num_classes=4).to(device)
  
    
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Checkpoint not found: {CKPT_PATH}")
        print("Please check the checkpoint path!")
        return
    
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✓ Model loaded from: {CKPT_PATH}")
    
    print("\n" + "="*70)
    print("Starting inference...")
    print("="*70)
    
    with torch.no_grad():
        for idx, (sar, opt, label, filename) in enumerate(tqdm(val_loader, desc='Inference')):
            sar, opt = sar.to(device), opt.to(device)
            
            # 前向推理
            outputs = model(sar, opt)
            if isinstance(outputs, dict):
                pred_map = outputs['damage']
            else:
                pred_map = outputs
            
            # 获取预测结果
            pred = torch.argmax(pred_map, dim=1).cpu().numpy()[0]  # shape: (H, W)
            
            # 获取文件名
            if isinstance(filename, (list, tuple)):
                fname = filename[0]
            else:
                fname = filename
            
            # 去掉扩展名
            base_name = os.path.splitext(fname)[0]
            
            # 保存raw格式 (保存为灰度图，像素值即为类别ID)
            raw_img = Image.fromarray(pred.astype(np.uint8), mode='L')
            raw_path = os.path.join(RAW_DIR, f"{base_name}.png")
            raw_img.save(raw_path)
            
            # 保存彩色格式
            color_img = colorize_prediction(pred)
            color_pil = Image.fromarray(color_img, mode='RGB')
            color_path = os.path.join(COLOR_DIR, f"{base_name}.png")
            color_pil.save(color_path)
    
    print("\n" + "="*70)
    print("🎉 Inference completed!")
    print(f"📊 Total samples: {len(val_dataset)}")
    print(f"💾 Raw results saved to: {RAW_DIR}")
    print(f"🎨 Color results saved to: {COLOR_DIR}")
    print("="*70)
    
    print("\nColor mapping:")
    print("  No damage (0): White [255, 255, 255]")
    print("  Intact (1): Green [70, 181, 121]")
    print("  Damaged (2): Orange [228, 189, 139]")
    print("  Destroyed (3): Red [182, 70, 69]")

if __name__ == '__main__':
    main()
