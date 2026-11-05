# train_baseline.py
# 基线模型训练脚本（YOLOv8n，无注意力模块）
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent  # 仓库根目录（即本脚本所在目录）

print("=" * 60)
print(" 基线模型训练（YOLOv8n，无注意力模块）")
print("=" * 60)

# 使用原始 YOLOv8n 模型
model = YOLO('yolov8n.pt')

# 训练参数（与改进模型保持一致）
results = model.train(
    data=str(ROOT / 'dataset.yaml'),
    epochs=200,              # 与改进模型相同
    batch=4,
    imgsz=640,
    device='cpu',
    workers=0,

    # ========== 几何变换（与改进模型相同）==========
    degrees=90,              # 旋转 ±90°
    scale=0.5,               # 缩放 0.5-1.5倍
    shear=15,                # 剪切 ±15°
    translate=0.1,           # 平移 ±10%
    fliplr=0.5,              # 水平翻转 50%
    flipud=0.3,              # 垂直翻转 30%
    # ==============================================

    # ========== 色彩变换（与改进模型相同）==========
    hsv_h=0.02,              # 色调变化
    hsv_s=0.8,               # 饱和度变化
    hsv_v=0.4,               # 明度变化
    # ==============================================

    # 学习率设置
    lr0=0.0005,              # 初始学习率
    lrf=0.0005,              # 最终学习率
    warmup_epochs=5,         # 预热轮数
    warmup_momentum=0.8,     # 预热动量
    warmup_bias_lr=0.05,     # 预热偏置学习率

    # 优化器设置
    optimizer='AdamW',
    momentum=0.937,          # 动量参数
    weight_decay=0.0005,     # 权重衰减
    # 损失权重（与改进模型相同）
    cls=0.8,                 # 分类损失权重
    box=7.5,                 # 边界框损失
    dfl=1.5,                 # 分布焦点损失

    # 早停设置
    patience=50,             # 60轮不提升就停止

    # 保存设置
    name='baseline_v3',      # 基线模型名称
    exist_ok=True,
    verbose=True
)

print("✅ 基线模型训练完成！")
print(f"模型保存在: runs/detect/baseline_v3/weights/best.pt")