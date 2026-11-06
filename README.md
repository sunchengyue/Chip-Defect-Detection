# 基于机器视觉的芯片表面缺陷检测（YOLOv8n-SimAM）

本项目为结课论文配套的代码与模型权重，基于 YOLOv8n 改进，用于芯片表面四类缺陷（划痕、破损、颗粒、污渍）的自动检测。

在骨干网络 P3/8 特征层后插入无参数注意力模块 **SimAM**，在不增加模型参数量的前提下增强细粒度缺陷的特征响应，mAP50 由 0.914 提升至 0.926。

## 改进思路

- **SimAM 注意力**：基于能量函数为每个神经元估计重要性权重，直接作用于特征图，不引入任何额外参数，因此模型体积与推理速度几乎不变。
- **插入位置**：骨干网络第 5 层（P3/8，80×80×256）C2f 之后、下采样之前。该尺度与划痕等细长小目标最匹配。
- **公平对比**：基线模型与改进模型均从 COCO 预训练权重初始化，训练参数（数据增强、学习率、优化器、损失权重）保持一致，只改变网络结构。

## 实验结果

| 模型 | mAP50 | 精确率 | 召回率 | F1 |
|---|---|---|---|---|
| YOLOv8n（基线） | 0.914 | 0.905 | 0.859 | 0.880 |
| YOLOv8n-SimAM（改进） | **0.926** | **0.962** | **0.869** | **0.913** |
| 提升量 | +1.2% | +5.7% | +1.0% | +3.3% |

各类别 F1 对比：

| 缺陷类别 | 基线 F1 | 改进 F1 | 提升幅度 |
|---|---|---|---|
| 颗粒 particle | 0.920 | 0.948 | +3.0% |
| 破损 broken | 0.790 | 0.945 | +19.6% |
| 污渍 stain | 0.900 | 0.978 | +8.7% |
| 划痕 scratch | 0.650 | 0.761 | +17.1% |

模型规模：改进模型融合后约 301 万参数、8.1 GFLOPs，输入 640×640，输出 1×8×8400。

## 目录结构

```
Chip-Defect-Detection/
├── train_baseline.py          基线模型训练脚本（YOLOv8n）
├── train_simam.py             改进模型训练脚本（YOLOv8n-SimAM）
├── predict.py                 批量预测与可视化脚本
├── yolov8n_simam_middle.yaml  改进模型网络结构配置（SimAM 置于骨干第5层后）
├── dataset.yaml               数据集路径与类别配置
├── attention.py               SimAM 模块定义（加载 simam 权重需要）
├── requirements.txt           环境依赖（ultralytics 版本已固定）
├── datasets/chip_defect_detection/  10 张验证集样例（含标注，可直接跑 predict.py）
└── weights/
    ├── baseline/best.pt       基线模型最佳权重（mAP50=0.914）
    └── simam/best.pt          改进模型最佳权重（mAP50=0.926）
```

## 环境准备

```bash
pip install -r requirements.txt
```

`requirements.txt` 已固定 `ultralytics==8.4.37`（与训练时一致，保证权重可正常加载）。开发环境：Python 3.13，PyTorch 2.11.0（CPU 训练同样可复现）。

## 数据集准备

仓库自带 10 张验证集样例（含标注）在 `datasets/chip_defect_detection/`，可直接运行 `predict.py` 体验。完整训练集不随仓库分发，训练前需自行按 YOLO 格式整理数据放到同一目录：

```
datasets/chip_defect_detection/
├── images/train
├── images/val
├── labels/train
└── labels/val
```

`dataset.yaml` 已配置好 4 个类别：

```yaml
nc: 4
names: [scratch, broken, particle, stain]
```

## 使用说明

训练基线模型：

```bash
python train_baseline.py
```

训练改进模型：

```bash
python train_simam.py
```

批量预测并统计各类缺陷数量（结果图输出到 `baseline_results1/`）：

```bash
python predict.py
```

若只想用训练好的权重直接推理：

```python
from ultralytics import YOLO

model = YOLO('weights/simam/best.pt')
results = model('chip.jpg', conf=0.3)
results[0].save('out.jpg')
```

权重文件自带网络结构与类别名，无需再加载 yaml 配置。

**注意**：`weights/simam/best.pt` 包含自定义的 SimAM 模块，加载前需将仓库根目录的 `attention.py` 复制到所装 ultralytics 包的 `nn/modules/` 目录下（即 `ultralytics/nn/modules/attention.py`），否则反序列化时会报 `ModuleNotFoundError`。基线权重无此要求。

## 引用

SimAM 模块来自：

```bibtex
@inproceedings{yang2021simam,
  title={SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks},
  author={Yang, Lingxiao and Zhang, Ru-Yuan and Li, Lida and Xie, Xiaohua},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
  pages={11863--11874},
  year={2021}
}
```

基座模型为 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)（AGPL-3.0）。
