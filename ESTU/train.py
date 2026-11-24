import datetime
import os
import gc

# Suppress all warnings and verbose outputs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings

warnings.filterwarnings("ignore")

from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Import Sobel enhancement related modules
from nets.modified_unet import create_sobel_unet
from utils.sobel_loss import CombinedLoss
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch


# Silent seed setting
def silent_seed_everything(seed):
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Clean configuration display function
def clean_show_config(num_classes, backbone, model_path, input_shape, Init_Epoch, Freeze_Epoch, UnFreeze_Epoch,
                      Freeze_batch_size, Unfreeze_batch_size, Freeze_Train, Init_lr, Min_lr, optimizer_type,
                      momentum, lr_decay_type, save_period, save_dir, num_workers, num_train, num_val,
                      data_augmentation, edge_loss_weight):
    print("=" * 80)
    print("|{:>35} | {:>40}|".format("Parameter", "Value"))
    print("=" * 80)
    print("|{:>35} | {:>40}|".format("Model Type", "Sobel-Enhanced U-Net"))
    print("|{:>35} | {:>40}|".format("num_classes", num_classes))
    print("|{:>35} | {:>40}|".format("backbone", backbone))
    print("|{:>35} | {:>40}|".format("model_path", model_path))
    print("|{:>35} | {:>40}|".format("input_shape", str(input_shape)))
    print("|{:>35} | {:>40}|".format("edge_loss_weight", edge_loss_weight))
    print("|{:>35} | {:>40}|".format("Init_Epoch", Init_Epoch))
    print("|{:>35} | {:>40}|".format("Freeze_Epoch", Freeze_Epoch))
    print("|{:>35} | {:>40}|".format("UnFreeze_Epoch", UnFreeze_Epoch))
    print("|{:>35} | {:>40}|".format("Freeze_batch_size", Freeze_batch_size))
    print("|{:>35} | {:>40}|".format("Unfreeze_batch_size", Unfreeze_batch_size))
    print("|{:>35} | {:>40}|".format("Freeze_Train", Freeze_Train))
    print("|{:>35} | {:>40}|".format("Init_lr", Init_lr))
    print("|{:>35} | {:>40}|".format("Min_lr", f"{Min_lr:.2e}"))
    print("|{:>35} | {:>40}|".format("optimizer_type", optimizer_type))
    print("|{:>35} | {:>40}|".format("momentum", momentum))
    print("|{:>35} | {:>40}|".format("lr_decay_type", lr_decay_type))
    print("|{:>35} | {:>40}|".format("save_period", save_period))
    print("|{:>35} | {:>40}|".format("save_dir", save_dir))
    print("|{:>35} | {:>40}|".format("num_workers", num_workers))
    print("|{:>35} | {:>40}|".format("num_train", num_train))
    print("|{:>35} | {:>40}|".format("num_val", num_val))
    print("|{:>35} | {:>40}|".format("data_augmentation", data_augmentation))
    print("=" * 80)


# Silent memory cleanup function
def clear_memory(verbose=False):
    """Clear GPU and CPU memory silently"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
    # Removed verbose printing to keep logs clean


# Safe loading of pretrained weights
def load_pretrained_weights_safe(model, model_path):
    """Safely load pretrained weights, skip parameters with size mismatch"""
    if model_path != '':
        print(f'Loading pretrained weights: {model_path}')
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path, map_location=device)

            filtered_dict = {}
            skipped_params = []

            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                    else:
                        skipped_params.append(k)
                        print(f"Skipping mismatched parameter: {k}")

            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=False)
            print(
                f"Successfully loaded {len(filtered_dict)} parameters, skipped {len(skipped_params)} mismatched parameters")

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Continuing with random initialization")
    return model


# Silent memory-efficient training wrapper
def memory_efficient_fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                                   epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss,
                                   focal_loss,
                                   cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank):
    """
    Wrapper for fit_one_epoch with silent memory management
    """

    # Create custom data iterators with silent memory management
    def silent_memory_managed_dataloader(dataloader, clear_interval=50):
        for i, batch in enumerate(dataloader):
            yield batch
            # Silent memory cleanup - every 50 iterations (no print messages)
            if (i + 1) % clear_interval == 0:
                torch.cuda.empty_cache()
                if (i + 1) % 200 == 0:  # More thorough cleanup every 200 iterations
                    gc.collect()

    # Patch the original dataloaders
    original_gen = gen
    original_gen_val = gen_val

    # Create silent memory-managed versions
    gen_memory_managed = silent_memory_managed_dataloader(original_gen, clear_interval=50)
    gen_val_memory_managed = silent_memory_managed_dataloader(original_gen_val, clear_interval=30)

    try:
        # Call original function with silent memory management
        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                      epoch_step, epoch_step_val, gen_memory_managed, gen_val_memory_managed, UnFreeze_Epoch, Cuda,
                      dice_loss, focal_loss,
                      cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)
    except Exception as e:
        print(f"Error in memory_efficient_fit_one_epoch: {e}")
        # Fallback to original function
        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                      epoch_step, epoch_step_val, original_gen, original_gen_val, UnFreeze_Epoch, Cuda, dice_loss,
                      focal_loss,
                      cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)


# Memory-efficient F1 score calculation
def calculate_f1_score(model, val_loader, device, num_classes):
    """Calculate F1 score on validation set with silent memory optimization"""
    model.eval()

    # Use minimal samples to reduce memory usage
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if batch_idx >= 3:  # Reduced from 10 to 3 samples
                break

            # Handle different batch formats
            if len(batch_data) == 3:
                images, targets, labels = batch_data
            else:
                images, targets = batch_data

            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            if hasattr(model, 'module'):
                seg_outputs, _ = model.module(images)
            else:
                seg_outputs, _ = model(images)

            # Get predictions and move to CPU immediately
            preds = torch.argmax(seg_outputs, dim=1).cpu().numpy()
            targets_cpu = targets.cpu().numpy()

            # Sample only part of the data to reduce memory
            sample_size = min(1000, preds.size)  # Limit to 1000 pixels
            indices = np.random.choice(preds.size, sample_size, replace=False)

            all_preds.extend(preds.flatten()[indices])
            all_targets.extend(targets_cpu.flatten()[indices])

            # Clear GPU memory silently
            del seg_outputs, preds, targets_cpu
            torch.cuda.empty_cache()

    # Calculate F1 score
    try:
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    except Exception as e:
        print(f"F1 calculation error: {e}")
        f1 = 0.0

    model.train()
    clear_memory()  # Silent memory clear
    return f1


# Enhanced version: Customizable for different paper needs
def save_edge_visualization(images, seg_preds, edge_preds, targets, save_path, epoch,
                            include_titles=False, include_axes=False, high_dpi=True):
    """
    Save edge prediction visualization with customizable options for paper use

    Args:
        images: Input images
        seg_preds: Segmentation predictions
        edge_preds: Edge predictions
        targets: Ground truth targets
        save_path: Path to save the visualization
        epoch: Current epoch number
        include_titles: Whether to include subplot titles
        include_axes: Whether to show axes
        high_dpi: Whether to use high DPI for paper quality
    """
    try:
        import matplotlib.pyplot as plt

        # Select first 2 samples for visualization (reduced from 4)
        batch_size = min(2, images.shape[0])

        # Adjust figure size for paper - you can modify this
        fig_width = 16 if include_titles else 15
        fig_height = 4 * batch_size if include_titles else 3.5 * batch_size

        fig, axes = plt.subplots(batch_size, 4, figsize=(fig_width, fig_height))

        if batch_size == 1:
            axes = axes.reshape(1, -1)

        # Column labels (only if titles are included)
        column_titles = ['Original Image', 'Ground Truth', 'Segmentation Prediction', 'Edge Prediction']

        for i in range(batch_size):
            # Original image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)
            axes[i, 0].imshow(img)
            if include_titles and i == 0:  # Only add title to first row
                axes[i, 0].set_title(column_titles[0], fontsize=12, pad=10)
            if not include_axes:
                axes[i, 0].axis('off')

            # Ground truth
            target = targets[i].cpu().numpy()
            axes[i, 1].imshow(target, cmap='viridis')
            if include_titles and i == 0:
                axes[i, 1].set_title(column_titles[1], fontsize=12, pad=10)
            if not include_axes:
                axes[i, 1].axis('off')

            # Segmentation prediction
            seg_pred = torch.argmax(seg_preds[i], dim=0).cpu().numpy()
            axes[i, 2].imshow(seg_pred, cmap='viridis')
            if include_titles and i == 0:
                axes[i, 2].set_title(column_titles[2], fontsize=12, pad=10)
            if not include_axes:
                axes[i, 2].axis('off')

            # Edge prediction
            edge_pred = edge_preds[i, 0].cpu().detach().numpy()
            axes[i, 3].imshow(edge_pred, cmap='gray')
            if include_titles and i == 0:
                axes[i, 3].set_title(column_titles[3], fontsize=12, pad=10)
            if not include_axes:
                axes[i, 3].axis('off')

        # Layout and save settings
        pad = 0.5 if include_titles else 0.1
        plt.tight_layout(pad=pad)

        dpi = 300 if high_dpi else 150
        pad_inches = 0.1 if include_titles else 0.02

        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)
        plt.close()
        # Silent save - no print message

    except Exception as e:
        print(f"Error saving edge visualization: {e}")
    finally:
        clear_memory()  # Silent memory clear


# Create a custom context manager to suppress specific prints
class SuppressSpecificPrints:
    def __init__(self, keywords_to_suppress):
        self.keywords_to_suppress = keywords_to_suppress
        self.original_stdout = None

    def __enter__(self):
        import sys
        from io import StringIO

        class FilteredOutput:
            def __init__(self, original_stdout, keywords):
                self.original_stdout = original_stdout
                self.keywords = keywords
                self.buffer = ""

            def write(self, text):
                # Check if any suppressed keywords are in the text
                if any(keyword in text for keyword in self.keywords):
                    return  # Suppress this output
                else:
                    self.original_stdout.write(text)

            def flush(self):
                self.original_stdout.flush()

        self.original_stdout = sys.stdout
        sys.stdout = FilteredOutput(sys.stdout, self.keywords_to_suppress)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        sys.stdout = self.original_stdout


if __name__ == "__main__":
    # ============ Configuration Parameters ============
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = True

    num_classes = 4
    backbone = "swin_transformer_v2_large"
    pretrained = True
    model_path = os.environ.get("PRETRAINED_WEIGHTS", os.path.join(os.path.dirname(__file__), "model_data/swinv2_large_patch4_window12_192_22k.pth"))

    input_shape = [256, 256]

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 20  # Restored original batch size
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 25  # Restored original batch size
    Freeze_Train = True

    # Adjust learning rate for edge-enhanced model
    Init_lr = 8e-5  # Slightly reduced
    Min_lr = Init_lr * 0.01
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 0.05
    lr_decay_type = 'cos'

    save_period = 5
    save_dir = 'logs_sobel_enhanced'
    eval_flag = True
    eval_period = 5

    VOCdevkit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "VOCdevkit"))
    dice_loss = True
    focal_loss = True
    cls_weights = np.array([1.0, 1.2, 1.5, 1.8], np.float32)
    num_workers = 4  # Restored original value

    # ============ Sobel Enhancement Specific Parameters ============
    edge_loss_weight = 0.3  # Edge loss weight
    edge_visualization = True  # Whether to save edge prediction visualization
    edge_viz_period = 10  # Edge visualization save period
    data_augmentation = False

    # Set random seed
    silent_seed_everything(seed)

    # Device configuration
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    print(f"Starting Sobel Enhanced U-Net Training")
    print(f"Using device: {device}")

    # Silent initial memory cleanup
    clear_memory()

    # ============ Create Sobel Enhanced Model ============
    # Suppress model creation prints
    with SuppressSpecificPrints(['TIMM available']):
        try:
            print("Creating Sobel Enhanced U-Net model...")
            model = create_sobel_unet(
                num_classes=num_classes,
                pretrained=pretrained,
                backbone=backbone,
                input_shape=input_shape
            ).train()

            # Load pretrained weights to backbone
            if model_path != '':
                model.backbone = load_pretrained_weights_safe(model.backbone, model_path)

        except Exception as e:
            print(f"Error creating model: {e}")

    print("Sobel Enhanced U-Net model created successfully")

    # Silent memory cleanup after model creation
    clear_memory()

    # ============ Setup Loss Function ============
    loss_function = CombinedLoss(
        num_classes=num_classes,
        dice_loss=dice_loss,
        focal_loss=focal_loss,
        cls_weights=torch.FloatTensor(cls_weights),
        edge_weight=edge_loss_weight
    )
    print(f"Loss function setup complete, edge loss weight: {edge_loss_weight}")

    # Loss history recording
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)

        # Create edge visualization save directory
        edge_viz_dir = os.path.join(log_dir, "edge_visualizations")
        os.makedirs(edge_viz_dir, exist_ok=True)
    else:
        loss_history = None
        edge_viz_dir = None

    # Mixed precision training
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    # Multi-GPU setup
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # Silent memory cleanup after model setup
    clear_memory()

    # Load dataset
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # Display configuration
    if local_rank == 0:
        clean_show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path,
            input_shape=input_shape, Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch,
            UnFreeze_Epoch=UnFreeze_Epoch, Freeze_batch_size=Freeze_batch_size,
            Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type,
            momentum=momentum, lr_decay_type=lr_decay_type, save_period=save_period,
            save_dir=save_dir, num_workers=num_workers, num_train=num_train,
            num_val=num_val, data_augmentation=data_augmentation,
            edge_loss_weight=edge_loss_weight
        )

    # Training setup
    UnFreeze_flag = False

    if Freeze_Train:
        model.freeze_backbone()
        print("Backbone frozen")

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    # Learning rate calculation
    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 1e-1
    lr_limit_min = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # Optimizer
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                              weight_decay=weight_decay)

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    # Calculate steps
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset too small for training")

    # Create datasets - suppress dataset creation prints
    with SuppressSpecificPrints(['TIMM available']):
        try:
            train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path, data_augmentation)
            val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path, data_augmentation)
        except Exception as e:
            print(f"Error creating datasets: {e}")

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    # Data loaders
    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

    # Evaluation callback
    if local_rank == 0:
        eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
                                     eval_flag=eval_flag, period=eval_period)
    else:
        eval_callback = None

    print("Starting Sobel Enhanced U-Net Training...")

    # Silent memory cleanup before training
    clear_memory()

    # ============ Training Loop ============
    # Suppress training prints that are memory-related
    memory_suppress_keywords = [
        'Deep memory clear at training iteration',
        'Memory cleared',
        'TIMM available'
    ]

    with SuppressSpecificPrints(memory_suppress_keywords):
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # Silent memory cleanup at start of each epoch
            clear_memory()

            # Unfreezing process
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 1e-1
                lr_limit_min = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.unfreeze_backbone()
                print(f"Epoch {epoch}: Backbone unfrozen")

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate,
                                 sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate,
                                     sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True
                clear_memory()  # Silent memory clear after unfreezing

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # Training with error handling and silent memory management
            try:
                # Use memory-efficient wrapper with silent memory management
                memory_efficient_fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                                               epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda,
                                               dice_loss,
                                               focal_loss,
                                               cls_weights, num_classes, fp16, scaler, save_period, save_dir,
                                               local_rank)
            except (torch.cuda.OutOfMemoryError, MemoryError) as e:
                print(f"Memory error at epoch {epoch}: {e}")
                print("Clearing memory and continuing...")
                clear_memory()
                # Try to continue training
                continue
            except Exception as e:
                print(f"Unexpected error at epoch {epoch}: {e}")
                clear_memory()
                continue

            # Edge prediction visualization
            if local_rank == 0 and edge_visualization and (epoch + 1) % edge_viz_period == 0:
                try:
                    model.eval()
                    with torch.no_grad():
                        # Get a validation batch
                        val_batch = next(iter(gen_val))
                        if len(val_batch) == 3:
                            images, targets, labels = val_batch
                        else:
                            images, targets = val_batch

                        images = images.to(device)
                        targets = targets.to(device)

                        # Forward pass
                        seg_outputs, edge_outputs = model(images)

                        # Save visualization
                        viz_path = os.path.join(edge_viz_dir, f"epoch_{epoch + 1}_edge_predictions.png")
                        save_edge_visualization(images, seg_outputs, edge_outputs, targets, viz_path, epoch + 1)

                    model.train()
                except Exception as e:
                    print(f"Error in edge visualization: {e}")
                finally:
                    clear_memory()

            # Silent enhanced memory cleanup - only print every 10 epochs to reduce log spam
            if (epoch + 1) % 10 == 0:
                clear_memory()
                # Only print memory status every 10 epochs instead of every 5
                print(f"Epoch {epoch + 1}: Periodic memory cleanup completed")

            if distributed:
                dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()
        print("Sobel Enhanced U-Net training completed!")
        print(f"Model save path: {save_dir}")
        if edge_visualization:
            print(f"Edge visualization save path: {edge_viz_dir}")

    # Final silent memory cleanup
    clear_memory()