# predict_all.py
# 批量预测脚本：使用训练好的模型对验证集图片进行缺陷检测，并统计结果

from ultralytics import YOLO  # 导入YOLO类，用于加载模型和进行预测
import cv2  # 导入OpenCV库，用于图像读取、处理和保存
import os  # 导入操作系统模块，用于文件和目录操作
import time  # 导入时间模块，用于计算预测耗时
from collections import Counter  # 导入Counter类，用于统计各类缺陷数量
from pathlib import Path  # 导入路径处理模块

ROOT = Path(__file__).resolve().parent  # 仓库根目录（即本脚本所在目录）

print("=" * 60)  # 打印60个等号作为分隔线
print(" 开始批量预测芯片缺陷")  # 打印预测任务标题
print("=" * 60)  # 打印分隔线

# ==================== 可调整参数 ====================
CONF_THRESHOLD = 0.3  # 置信度阈值：模型认为检测正确的置信度下限（0.3=30%），调低可检测更多但可能增加误检
IOU_THRESHOLD = 0.45  # IoU（交并比）阈值：去除重叠框的阈值，值越小去重越严格
# ===================================================

# 打印当前使用的参数设置
print(f" 当前设置: 置信度阈值={CONF_THRESHOLD}, IoU阈值={IOU_THRESHOLD}")

# 模型路径列表（可添加多个备选路径，自动查找第一个存在的模型）
model_paths = [
    str(ROOT / 'weights' / 'baseline' / 'best.pt')  # 基线模型的最佳权重文件路径
]

model_path = None  # 初始化模型路径变量
for path in model_paths:  # 遍历备选路径列表
    if os.path.exists(path):  # 检查文件是否存在
        model_path = path  # 找到存在的模型文件
        print(f" 找到模型: {model_path}")  # 打印找到的模型路径
        break  # 找到后退出循环

if model_path is None:  # 如果没有找到任何模型文件
    print(f"未找到模型文件，请检查路径")  # 打印错误提示
    exit()  # 退出程序

model = YOLO(model_path)  # 加载YOLO模型（从指定路径加载权重）
print(" 模型加载成功")  # 打印模型加载成功提示

# 验证集图片路径
val_dir = str(ROOT / 'datasets' / 'chip_defect_detection' / 'images' / 'val')

if not os.path.exists(val_dir):  # 检查验证集目录是否存在
    print(f" 验证集目录不存在: {val_dir}")  # 打印错误提示
    exit()  # 退出程序

# 获取验证集目录中的所有图片文件
val_images = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"找到 {len(val_images)} 张验证图片")  # 打印找到的图片数量

# 创建输出文件夹
output_dir = 'baseline_results1'  # 输出目录名称
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（如果已存在则不报错）

# 统计变量初始化（数量统计）
total_defects = 0  # 总缺陷数
defect_counts = Counter()  # 各类缺陷计数器（自动初始化为0）
images_with_defects = 0  # 有缺陷的图片数量
images_without_defects = 0  # 无缺陷的图片数量
scratch_detected = 0  # 划痕检测数量
broken_detected = 0  # 破损检测数量
particle_detected = 0  # 颗粒检测数量
stain_detected = 0  # 污渍检测数量

# 记录开始时间
start_time = time.time()  # 获取当前时间戳，用于计算总处理时间

print("\n 开始批量预测...")  # 打印开始预测提示
print("=" * 60)  # 打印分隔线

# 遍历每张验证图片进行预测
for i, img_name in enumerate(val_images):  # enumerate返回索引和文件名
    img_path = os.path.join(val_dir, img_name)  # 拼接图片完整路径

    # 使用模型进行预测，传入置信度阈值和IoU阈值
    results = model(img_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)

    # 获取带检测框的标注图像
    result_img = results[0].plot()  # plot()函数将检测框绘制在图像上
    output_path = os.path.join(output_dir, f'pred_{img_name}')  # 拼接输出路径（文件名加pred_前缀）
    cv2.imwrite(output_path, result_img)  # 保存标注后的图像

    boxes = results[0].boxes  # 获取检测到的边界框列表
    if boxes is not None and len(boxes) > 0:  # 如果检测到缺陷（有边界框）
        num_defects = len(boxes)  # 当前图片的缺陷数量
        total_defects += num_defects  # 累加到总缺陷数
        images_with_defects += 1  # 有缺陷的图片数加1

        # 遍历每个检测到的缺陷
        for box in boxes:
            cls = int(box.cls[0])  # 获取缺陷类别ID（0=划痕,1=破损,2=颗粒,3=污渍）
            class_names = ['scratch', 'broken', 'particle', 'stain']  # 类别名称列表
            defect_counts[class_names[cls]] += 1  # 对应类别计数加1

            # 分别统计各类别检测数量（用于额外统计）
            if cls == 0:
                scratch_detected += 1  # 划痕计数加1
            elif cls == 1:
                broken_detected += 1  # 破损计数加1
            elif cls == 2:
                particle_detected += 1  # 颗粒计数加1
            elif cls == 3:
                stain_detected += 1  # 污渍计数加1

        # 每处理10张图片打印一次进度
        if (i + 1) % 10 == 0:
            print(f"  已处理: {i + 1}/{len(val_images)} 张")
    else:  # 如果没有检测到缺陷
        images_without_defects += 1  # 无缺陷图片数加1

end_time = time.time()  # 获取结束时间戳

# ========== 打印统计结果 ==========
print("\n" + "=" * 60)  # 打印分隔线
print(" 预测统计结果")  # 打印统计标题
print("=" * 60)  # 打印分隔线
print(f"总图片数: {len(val_images)}")  # 打印总图片数量
print(f"有缺陷的图片: {images_with_defects} 张")  # 打印有缺陷的图片数量
print(f"无缺陷的图片: {images_without_defects} 张")  # 打印无缺陷的图片数量
print(f"总缺陷数: {total_defects} 个")  # 打印检测到的总缺陷数
print(f"平均每张图片缺陷数: {total_defects / len(val_images):.2f}")  # 打印平均每张图片的缺陷数

print("\n 各类缺陷统计:")  # 打印各类缺陷统计标题
print(f"  scratch (划痕): {defect_counts.get('scratch', 0)} 个")  # 打印划痕数量
print(f"  broken (破损): {defect_counts.get('broken', 0)} 个")  # 打印破损数量
print(f"  particle (颗粒): {defect_counts.get('particle', 0)} 个")  # 打印颗粒数量
print(f"  stain (污渍): {defect_counts.get('stain', 0)} 个")  # 打印污渍数量

print(f"\n 总处理时间: {end_time - start_time:.2f} 秒")  # 打印总处理时间
print(f"平均每张图片: {(end_time - start_time) / len(val_images) * 1000:.2f} 毫秒")  # 打印平均单张图片处理时间
print(f"\n 结果图片保存在: {output_dir}/")  # 打印结果保存路径
print("=" * 60)  # 打印分隔线

print("\n 提示:")  # 打印参数调整提示
print(f"  - 当前置信度阈值: {CONF_THRESHOLD}")  # 显示当前置信度阈值
print(f"  - 如需检测更多缺陷，可将阈值调低到 0.2")  # 提示如何提高召回率
print(f"  - 如需更精确的结果，可将阈值调高到 0.5")  # 提示如何提高精确率