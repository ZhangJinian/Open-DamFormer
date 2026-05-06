"""DamFormer 完整训练（GPU 1）"""
import os
import sys
sys.path.insert(0, '/media/lzd/0A5CED4894A8F8FB/zjn/DisasterDetection/DamFormer')
sys.path.insert(0, '/media/lzd/0A5CED4894A8F8FB/zjn/DisasterDetection/FusionDamNet')

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import json
from datetime import datetime

from damformer import DamFormer,MyDamFormer
from dataset import FusionDamDataset
from loss import FusionDamLoss
from metrics import compute_confusion_matrix, compute_metrics_from_confmat

# 指定使用GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def evaluate(model, dataloader, device, num_classes=4):
    model.eval()
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total_loss = 0.0
    batch_count = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    with torch.no_grad():
        for sar, opt, label, _ in tqdm(dataloader, desc='Evaluating', leave=False):
            sar, opt, label = sar.to(device), opt.to(device), label.to(device)
            outputs = model(sar, opt)
            if isinstance(outputs, dict):
                pred_map = outputs['damage']
            else:
                pred_map = outputs
            loss = criterion(pred_map, label)
            total_loss += loss.item()
            batch_count += 1
            
            pred = torch.argmax(pred_map, dim=1)
            conf_mat += compute_confusion_matrix(pred.cpu(), label.cpu(), num_classes=num_classes)
    
    metrics = compute_metrics_from_confmat(conf_mat)
    return metrics, total_loss / max(batch_count, 1)

def main():
    print("="*70)
    print("DamFormer 完整训练（GPU 1）")
    print("="*70)
    
    TRAIN_PATH = '/media/lzd/0A5CED4894A8F8FB/zjn/DisasterDetection/BRIGHT'
    SAVE_DIR = '/media/lzd/0A5CED4894A8F8FB/zjn/DisasterDetection/DamFormer/results/MyDamformer'
    CKPT_DIR = os.path.join(SAVE_DIR, 'ckpt')
    LOG_DIR = os.path.join(SAVE_DIR, 'logs')
    
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    print("✓ Loading BRIGHT dataset...")
    dataset = FusionDamDataset(TRAIN_PATH, split='train', use_splitfile=False, img_size=512)
    
    print(f"  Total samples: {len(dataset)}")
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1, pin_memory=True)
    
    # model = DamFormer(in_channels=3, num_classes=4).to(device)
    model = MyDamFormer(in_channels=3, num_classes=4).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    
    print("\n" + "="*70)
    print("Starting full training (100 epochs)...")
    print("="*70)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': [],
        'val_oa': []
    }
    best_miou = 0.0
    
    for epoch in range(1, 101):
        print(f"\n[Epoch {epoch}/100]")
        print("-" * 70)
        
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for sar, opt, label, _ in tqdm(train_loader, desc='Training', leave=False):
            sar, opt, label = sar.to(device), opt.to(device), label.to(device)
            
            optimizer.zero_grad()
            outputs = model(sar, opt)
            if isinstance(outputs, dict):
                pred_map = outputs['damage']
            else:
                pred_map = outputs
            loss = criterion(pred_map, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        train_loss = epoch_loss / max(batch_count, 1)
        history['train_loss'].append(train_loss)
        scheduler.step()
        
        if epoch % 1 == 0:
            print("Evaluating...")
            metrics, val_loss = evaluate(model, test_loader, device)
            history['val_loss'].append(val_loss)
            
            miou = metrics['iou'][1:].mean() if len(metrics['iou']) > 1 else 0.0
            oa = metrics['OA']
            history['val_miou'].append(float(miou))
            history['val_oa'].append(float(oa))
            
            print(f"  📊 Train Loss: {train_loss:.4f}")
            print(f"  📊 Val Loss:   {val_loss:.4f}")
            print(f"  📊 Val mIoU:   {miou:.4f}")
            print(f"  📊 Val OA:     {oa:.4f}")
            
            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'best.pth'))
                print(f"  ✅ Best model saved (mIoU: {best_miou:.4f})")
    
    print("\n" + "="*70)
    print("Generating training curves...")
    
    epochs_train = range(1, len(history['train_loss']) + 1)
    epochs_eval = range(10, len(history['val_loss']) * 10 + 1, 10)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(epochs_train, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_eval, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs_eval, history['val_miou'], 'g-', linewidth=2, marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title(f'Validation mIoU (Best: {best_miou:.4f})')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs_eval, history['val_oa'], 'purple', linewidth=2, marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('OA')
    axes[2].set_title('Validation OA')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'training_curves.png'), dpi=100, bbox_inches='tight')
    print(f"✓ Training curves saved")
    
    config_dict = {
        'timestamp': datetime.now().isoformat(),
        'epochs': 100,
        'batch_size': 2,
        'train_test_split': '80/20',
        'gpu_device': 'GPU 1',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'best_miou': float(best_miou),
        'final_val_oa': float(history['val_oa'][-1]) if history['val_oa'] else 0,
        'dataset': 'BRIGHT_full',
        'model': 'DamFormer',
        'image_size': 512,
        'total_samples': len(dataset),
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset)
    }
    
    with open(os.path.join(LOG_DIR, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Config saved")
    
    print("\n" + "="*70)
    print("🎉 Training completed!")
    print(f"📈 Best mIoU: {best_miou:.4f}")
    print(f"📊 Final OA: {history['val_oa'][-1]:.4f}" if history['val_oa'] else "")
    print(f"💾 Checkpoint: {CKPT_DIR}/best.pth")
    print("="*70)

if __name__ == '__main__':
    main()
