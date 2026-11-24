import random
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_everything(seed=11):
    pass


    print(f"🎲 设置随机种子: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


    os.environ['PYTHONHASHSEED'] = str(seed)

    print("✅ 随机种子设置完成，确保训练可重现")


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    pass


    image = image.astype(np.float32)
    image /= 255.0


    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # image = (image - mean) / std

    return image


def show_config(**kwargs):
    pass


    print('🔧 Swin Transformer UNet 配置:')
    print('=' * 70)
    print('|%25s | %40s|' % ('参数', '值'))
    print('=' * 70)
    for key, value in kwargs.items():

        if key == 'backbone' and 'swin' in str(value):
            value = f"🌟 {value}"
        elif key == 'num_classes':
            value = f"🎯 {value} (背景、活树、部分死亡树、死树)"
        elif key == 'input_shape':
            value = f"📐 {value}"
        elif 'batch_size' in key:
            value = f"📊 {value}"
        elif 'lr' in key:
            value = f"📈 {value}"

        print('|%25s | %40s|' % (str(key), str(value)))
    print('=' * 70)


def download_weights(backbone, model_dir="./model_data"):
    pass


    import os
    from torch.hub import load_state_dict_from_url

    download_urls = {
        'vgg': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',

        'swin_transformer_v2_base': None,
        'swin_transformer_v2_large': None,
        'swin_transformer_v2_small': None,
    }

    if backbone not in download_urls:
        print(f"⚠️ 不支持自动下载 {backbone} 的权重")
        return

    url = download_urls[backbone]
    if url is None:
        print(f"⚠️ {backbone} 需要手动下载预训练权重")
        print("请从官方渠道下载权重文件并放置在model_data目录下")
        return

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"📁 创建目录: {model_dir}")

    print(f"🔄 下载 {backbone} 预训练权重...")
    try:
        load_state_dict_from_url(url, model_dir)
        print(f"✅ {backbone} 权重下载完成")
    except Exception as e:
        print(f"❌ 权重下载失败: {e}")


def get_forest_class_colors():
    pass
    return {
        0: [0, 0, 0],
        1: [0, 255, 0],
        2: [255, 255, 0],
        3: [255, 0, 0]
    }


def get_forest_class_names():
    pass
    return {
        0: '背景',
        1: '活树',
        2: '部分死亡树',
        3: '死树'
    }


def visualize_forest_prediction(image, prediction, save_path=None, alpha=0.6):
    pass


    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))


    colors = get_forest_class_colors()
    h, w = prediction.shape
    colored_pred = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colors.items():
        mask = prediction == class_id
        colored_pred[mask] = color

    colored_pred_img = Image.fromarray(colored_pred)


    overlay = Image.blend(image, colored_pred_img, alpha)


    overlay_with_legend = add_forest_legend(overlay)

    if save_path:
        overlay_with_legend.save(save_path)
        print(f"💾 可视化结果保存到: {save_path}")

    return overlay_with_legend


def add_forest_legend(image):
    pass
    draw = ImageDraw.Draw(image)


    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = get_forest_class_colors()
    names = get_forest_class_names()


    x_start = image.width - 150
    y_start = 10

    for i, (class_id, color) in enumerate(colors.items()):
        y = y_start + i * 25


        draw.rectangle([x_start, y, x_start + 20, y + 20], fill=tuple(color))


        draw.text((x_start + 25, y + 2), names[class_id], fill=(255, 255, 255), font=font)

    return image


def calculate_forest_health_metrics(prediction):
    pass


    total_pixels = prediction.size


    background_pixels = np.sum(prediction == 0)
    live_tree_pixels = np.sum(prediction == 1)
    partial_dead_pixels = np.sum(prediction == 2)
    dead_tree_pixels = np.sum(prediction == 3)


    tree_pixels = total_pixels - background_pixels

    if tree_pixels == 0:
        return {
            'forest_coverage': 0.0,
            'live_tree_ratio': 0.0,
            'dead_tree_ratio': 0.0,
            'partial_dead_ratio': 0.0,
            'health_index': 0.0
        }


    forest_coverage = tree_pixels / total_pixels
    live_tree_ratio = live_tree_pixels / tree_pixels
    partial_dead_ratio = partial_dead_pixels / tree_pixels
    dead_tree_ratio = dead_tree_pixels / tree_pixels


    health_index = live_tree_ratio - 0.5 * partial_dead_ratio - dead_tree_ratio

    return {
        'forest_coverage': forest_coverage,
        'live_tree_ratio': live_tree_ratio,
        'partial_dead_ratio': partial_dead_ratio,
        'dead_tree_ratio': dead_tree_ratio,
        'health_index': max(0.0, health_index)
    }


def print_forest_health_report(metrics):
    pass
    print("🌲 林地健康度分析报告")
    print("=" * 50)
    print(f"🌲 森林覆盖率: {metrics['forest_coverage']:.2%}")
    print(f"🟢 活树比例: {metrics['live_tree_ratio']:.2%}")
    print(f"🟡 部分死亡树比例: {metrics['partial_dead_ratio']:.2%}")
    print(f"🔴 死树比例: {metrics['dead_tree_ratio']:.2%}")
    print(f"📊 健康度指数: {metrics['health_index']:.3f}")
    print("=" * 50)


    health_index = metrics['health_index']
    if health_index >= 0.8:
        grade = "A (优秀)"
        emoji = "🌟"
    elif health_index >= 0.6:
        grade = "B (良好)"
        emoji = "✅"
    elif health_index >= 0.4:
        grade = "C (一般)"
        emoji = "⚠️"
    elif health_index >= 0.2:
        grade = "D (较差)"
        emoji = "⚠️"
    else:
        grade = "E (很差)"
        emoji = "❌"

    print(f"{emoji} 整体健康等级: {grade}")


def log_gpu_memory():
    pass
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
        cached = torch.cuda.memory_reserved() / 1024 ** 3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB

        print(f"🖥️ GPU显存: {allocated:.1f}GB / {total:.1f}GB (已分配)")
        print(f"📦 GPU缓存: {cached:.1f}GB (已缓存)")

        return {
            'allocated': allocated,
            'cached': cached,
            'total': total,
            'usage_percent': allocated / total * 100
        }
    else:
        print("❌ 未检测到CUDA设备")
        return None


def time_function(func):
    pass

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ {func.__name__} 执行时间: {end_time - start_time:.2f}秒")
        return result

    return wrapper


def check_model_size(model):
    pass
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 模型分析:")
    print(f"  - 总参数量: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  - 可训练参数: {trainable_params:,} ({trainable_params / 1e6:.1f}M)")
    print(f"  - 冻结参数: {total_params - trainable_params:,}")


    model_size = total_params * 4 / 1024 / 1024
    print(f"  - 估算大小: {model_size:.1f}MB")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size
    }