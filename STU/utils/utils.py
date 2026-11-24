import random
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


#   Convert image to RGB format to prevent grayscale image errors during prediction.
#   Code only supports RGB image prediction, all other types will be converted to RGB

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


#   Resize input image

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


#   Get learning rate

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


#   🔧 Enhanced seed setting function
#   Ensures reproducibility for Swin Transformer training

def seed_everything(seed=11):
    """
    Set all random seeds to ensure experiment reproducibility
    Especially important for complex models like Swin Transformer
    """
    print(f"🎲 Setting random seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 🔧 Deterministic settings for Swin Transformer
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)

    print("✅ Random seed setup completed, ensuring reproducible training")


#   Set Dataloader seed

def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    """
    🔧 Improved image preprocessing
    Optimized for forest RGB imagery
    """
    image = image.astype(np.float32)
    image /= 255.0

    # Optional: Add ImageNet normalization (if using ImageNet pretrained)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # image = (image - mean) / std

    return image


def show_config(**kwargs):
    """
    🔧 Beautified configuration display function
    """
    print('🔧 Swin Transformer UNet Configuration:')
    print('=' * 70)
    print('|%25s | %40s|' % ('Parameter', 'Value'))
    print('=' * 70)
    for key, value in kwargs.items():
        # Special formatting for certain key parameters
        if key == 'backbone' and 'swin' in str(value):
            value = f"🌟 {value}"
        elif key == 'num_classes':
            value = f"🎯 {value} (background, live trees, partially dead trees, dead trees)"
        elif key == 'input_shape':
            value = f"📐 {value}"
        elif 'batch_size' in key:
            value = f"📊 {value}"
        elif 'lr' in key:
            value = f"📈 {value}"

        print('|%25s | %40s|' % (str(key), str(value)))
    print('=' * 70)


def download_weights(backbone, model_dir="./model_data"):
    """
    🔧 Improved weight download function
    Supports more backbone types
    """
    import os
    from torch.hub import load_state_dict_from_url

    download_urls = {
        'vgg': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
        # Can add Swin Transformer official weight download links
        'swin_transformer_v2_base': None,  # Requires manual download
        'swin_transformer_v2_large': None,  # Requires manual download
        'swin_transformer_v2_small': None,  # Requires manual download
    }

    if backbone not in download_urls:
        print(f"⚠️ Automatic download not supported for {backbone} weights")
        return

    url = download_urls[backbone]
    if url is None:
        print(f"⚠️ {backbone} requires manual download of pretrained weights")
        print("Please download weight files from official sources and place them in model_data directory")
        return

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"📁 Created directory: {model_dir}")

    print(f"🔄 Downloading {backbone} pretrained weights...")
    try:
        load_state_dict_from_url(url, model_dir)
        print(f"✅ {backbone} weights download completed")
    except Exception as e:
        print(f"❌ Weight download failed: {e}")


#   🌲 Forest segmentation specialized utility functions

def get_forest_class_colors():
    """Get class color mapping for forest segmentation"""
    return {
        0: [0, 0, 0],      # Background - Black
        1: [0, 255, 0],    # Live trees - Green
        2: [255, 255, 0],  # Partially dead trees - Yellow
        3: [255, 0, 0]     # Dead trees - Red
    }


def get_forest_class_names():
    """Get class names for forest segmentation"""
    return {
        0: 'Background',
        1: 'Live Trees',
        2: 'Partially Dead Trees',
        3: 'Dead Trees'
    }


def visualize_forest_prediction(image, prediction, save_path=None, alpha=0.6):
    """
    🌲 Visualize forest segmentation prediction results

    Args:
        image: Original RGB image (PIL Image or numpy array)
        prediction: Prediction result (numpy array, shape: [H, W])
        save_path: Save path
        alpha: Overlay transparency
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    # Create color mapping image
    colors = get_forest_class_colors()
    h, w = prediction.shape
    colored_pred = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colors.items():
        mask = prediction == class_id
        colored_pred[mask] = color

    colored_pred_img = Image.fromarray(colored_pred)

    # Overlay on original image
    overlay = Image.blend(image, colored_pred_img, alpha)

    # Add legend
    overlay_with_legend = add_forest_legend(overlay)

    if save_path:
        overlay_with_legend.save(save_path)
        print(f"💾 Visualization result saved to: {save_path}")

    return overlay_with_legend


def add_forest_legend(image):
    """Add legend to forest segmentation results"""
    draw = ImageDraw.Draw(image)

    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = get_forest_class_colors()
    names = get_forest_class_names()

    # Add legend in top-right corner of image
    x_start = image.width - 200
    y_start = 10

    for i, (class_id, color) in enumerate(colors.items()):
        y = y_start + i * 25

        # Draw color block
        draw.rectangle([x_start, y, x_start + 20, y + 20], fill=tuple(color))

        # Draw text
        draw.text((x_start + 25, y + 2), names[class_id], fill=(255, 255, 255), font=font)

    return image


def calculate_forest_health_metrics(prediction):
    """
    🌲 Calculate forest health metrics

    Args:
        prediction: Prediction result (numpy array)

    Returns:
        dict: Contains various health metrics
    """
    total_pixels = prediction.size

    # Calculate pixel count for each class
    background_pixels = np.sum(prediction == 0)
    live_tree_pixels = np.sum(prediction == 1)
    partial_dead_pixels = np.sum(prediction == 2)
    dead_tree_pixels = np.sum(prediction == 3)

    # Only consider tree pixels (excluding background)
    tree_pixels = total_pixels - background_pixels

    if tree_pixels == 0:
        return {
            'forest_coverage': 0.0,
            'live_tree_ratio': 0.0,
            'dead_tree_ratio': 0.0,
            'partial_dead_ratio': 0.0,
            'health_index': 0.0
        }

    # Calculate metrics
    forest_coverage = tree_pixels / total_pixels
    live_tree_ratio = live_tree_pixels / tree_pixels
    partial_dead_ratio = partial_dead_pixels / tree_pixels
    dead_tree_ratio = dead_tree_pixels / tree_pixels

    # Health index: live tree ratio - 0.5*partially dead ratio - dead tree ratio
    health_index = live_tree_ratio - 0.5 * partial_dead_ratio - dead_tree_ratio

    return {
        'forest_coverage': forest_coverage,
        'live_tree_ratio': live_tree_ratio,
        'partial_dead_ratio': partial_dead_ratio,
        'dead_tree_ratio': dead_tree_ratio,
        'health_index': max(0.0, health_index)  # Ensure non-negative
    }


def print_forest_health_report(metrics):
    """Print forest health report"""
    print("🌲 Forest Health Analysis Report")
    print("=" * 50)
    print(f"🌲 Forest Coverage: {metrics['forest_coverage']:.2%}")
    print(f"🟢 Live Trees Ratio: {metrics['live_tree_ratio']:.2%}")
    print(f"🟡 Partially Dead Ratio: {metrics['partial_dead_ratio']:.2%}")
    print(f"🔴 Dead Trees Ratio: {metrics['dead_tree_ratio']:.2%}")
    print(f"📊 Health Index: {metrics['health_index']:.3f}")
    print("=" * 50)

    # Health grade
    health_index = metrics['health_index']
    if health_index >= 0.8:
        grade = "A (Excellent)"
        emoji = "🌟"
    elif health_index >= 0.6:
        grade = "B (Good)"
        emoji = "✅"
    elif health_index >= 0.4:
        grade = "C (Fair)"
        emoji = "⚠️"
    elif health_index >= 0.2:
        grade = "D (Poor)"
        emoji = "⚠️"
    else:
        grade = "E (Very Poor)"
        emoji = "❌"

    print(f"{emoji} Overall Health Grade: {grade}")


#   🔧 Training monitoring tools

def log_gpu_memory():
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
        cached = torch.cuda.memory_reserved() / 1024 ** 3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB

        print(f"🖥️ GPU Memory: {allocated:.1f}GB / {total:.1f}GB (allocated)")
        print(f"📦 GPU Cache: {cached:.1f}GB (cached)")

        return {
            'allocated': allocated,
            'cached': cached,
            'total': total,
            'usage_percent': allocated / total * 100
        }
    else:
        print("❌ No CUDA device detected")
        return None


def time_function(func):
    """Decorator: measure function execution time"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ {func.__name__} execution time: {end_time - start_time:.2f}s")
        return result

    return wrapper


def check_model_size(model):
    """Check model size and parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 Model Analysis:")
    print(f"  - Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  - Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.1f}M)")
    print(f"  - Frozen parameters: {total_params - trainable_params:,}")

    # Estimate model size (MB)
    model_size = total_params * 4 / 1024 / 1024  # Assuming float32
    print(f"  - Estimated size: {model_size:.1f}MB")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size
    }