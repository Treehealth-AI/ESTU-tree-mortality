import datetime
import os
import gc
import psutil
import time
import glob
import shutil
import threading

# Suppress all warnings and verbose outputs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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

# Import STU related modules
from nets.stu_model import create_stu_model
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch


# ============ ADVANCED SYSTEM PROTECTION ============

class AdvancedSystemMonitor:
    """Advanced system monitoring with predictive protection"""

    def __init__(self):
        self.temperature_history = []
        self.memory_history = []
        self.cooldown_level = 0  # 0=normal, 1=warm, 2=hot, 3=critical
        self.consecutive_high_temp = 0
        self.forced_rest_count = 0

    def get_gpu_temperature(self):
        """Get GPU temperature information"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return temp
        except:
            return None

    def get_comprehensive_metrics(self):
        """Get comprehensive system metrics"""
        # CPU metrics
        memory = psutil.virtual_memory()
        cpu_memory_used = memory.used / (1024 ** 3)  # GB
        cpu_memory_total = memory.total / (1024 ** 3)  # GB
        cpu_memory_percent = memory.percent
        cpu_temp = None
        cpu_usage = psutil.cpu_percent(interval=1)

        # Try to get CPU temperature
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                cpu_temp = max([t.current for t in temps['coretemp']])
            elif 'k10temp' in temps:
                cpu_temp = temps['k10temp'][0].current
        except:
            pass

        # GPU metrics
        gpu_memory_used = 0
        gpu_memory_total = 0
        gpu_memory_percent = 0
        gpu_temp = self.get_gpu_temperature()
        gpu_usage = 0

        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total * 100) if gpu_memory_total > 0 else 0

            try:
                gpu_usage = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            except:
                pass

        metrics = {
            'cpu_used': cpu_memory_used,
            'cpu_total': cpu_memory_total,
            'cpu_percent': cpu_memory_percent,
            'cpu_temp': cpu_temp,
            'cpu_usage': cpu_usage,
            'gpu_used': gpu_memory_used,
            'gpu_total': gpu_memory_total,
            'gpu_percent': gpu_memory_percent,
            'gpu_temp': gpu_temp,
            'gpu_usage': gpu_usage,
            'timestamp': time.time()
        }

        # Update history
        self.temperature_history.append({
            'gpu_temp': gpu_temp,
            'cpu_temp': cpu_temp,
            'timestamp': time.time()
        })

        # Keep only last 10 readings
        if len(self.temperature_history) > 10:
            self.temperature_history = self.temperature_history[-10:]

        return metrics

    def predict_thermal_risk(self, current_metrics):
        """Predict thermal risk based on trends"""
        if len(self.temperature_history) < 3:
            return "LOW"

        recent_gpu_temps = [h['gpu_temp'] for h in self.temperature_history[-3:] if h['gpu_temp']]

        if not recent_gpu_temps:
            return "LOW"

        # Check temperature trend
        if len(recent_gpu_temps) >= 2:
            temp_increase = recent_gpu_temps[-1] - recent_gpu_temps[0]
            if temp_increase > 5:  # Rising fast
                return "HIGH"
            elif temp_increase > 2:  # Rising moderately
                return "MEDIUM"

        # Check current temperature
        current_gpu_temp = current_metrics.get('gpu_temp', 0) or 0
        if current_gpu_temp > 70:
            return "HIGH"
        elif current_gpu_temp > 65:
            return "MEDIUM"

        return "LOW"

    def adaptive_protection_check(self, current_metrics):
        """Advanced protection with adaptive thresholds"""
        # More conservative thresholds based on temperature trend
        risk_level = self.predict_thermal_risk(current_metrics)

        if risk_level == "HIGH":
            gpu_critical_temp = 68  # Very conservative
            cpu_critical_temp = 70
            memory_critical = 80
        elif risk_level == "MEDIUM":
            gpu_critical_temp = 72
            cpu_critical_temp = 75
            memory_critical = 85
        else:
            gpu_critical_temp = 75
            cpu_critical_temp = 80
            memory_critical = 90

        critical_issues = []
        warning_issues = []

        gpu_temp = current_metrics.get('gpu_temp')
        cpu_temp = current_metrics.get('cpu_temp')

        # Temperature checks
        if gpu_temp and gpu_temp > gpu_critical_temp:
            critical_issues.append(f"GPU temperature critical: {gpu_temp}°C > {gpu_critical_temp}°C")
            self.consecutive_high_temp += 1
        elif gpu_temp and gpu_temp > (gpu_critical_temp - 5):
            warning_issues.append(f"GPU temperature warning: {gpu_temp}°C")

        if cpu_temp and cpu_temp > cpu_critical_temp:
            critical_issues.append(f"CPU temperature critical: {cpu_temp}°C > {cpu_critical_temp}°C")

        # Memory checks
        if current_metrics['gpu_percent'] > memory_critical:
            critical_issues.append(f"GPU memory critical: {current_metrics['gpu_percent']:.1f}% > {memory_critical}%")

        if current_metrics['cpu_percent'] > memory_critical:
            critical_issues.append(f"CPU memory critical: {current_metrics['cpu_percent']:.1f}% > {memory_critical}%")

        # Update cooldown level
        if critical_issues:
            self.cooldown_level = min(3, self.cooldown_level + 1)
        elif warning_issues and self.cooldown_level > 0:
            pass  # Keep current level
        else:
            self.cooldown_level = max(0, self.cooldown_level - 1)

        return critical_issues, warning_issues, risk_level

    def get_recommended_cooldown(self):
        """Get recommended cooldown time based on system state"""
        base_cooldown = {
            0: 2,  # Normal: 2 seconds
            1: 15,  # Warm: 15 seconds
            2: 45,  # Hot: 45 seconds
            3: 120  # Critical: 2 minutes
        }

        cooldown_time = base_cooldown[self.cooldown_level]

        # Additional time if consecutive high temperatures
        if self.consecutive_high_temp > 3:
            cooldown_time += 60

        # Additional time based on forced rest count
        if self.forced_rest_count > 0:
            cooldown_time += min(self.forced_rest_count * 30, 180)

        return cooldown_time

    def reset_consecutive_counters(self):
        """Reset counters after successful operation"""
        self.consecutive_high_temp = max(0, self.consecutive_high_temp - 1)


def ultra_aggressive_memory_cleanup():
    """Ultra-aggressive memory cleanup with multiple strategies"""
    print("Performing ultra-aggressive memory cleanup...")

    # Python garbage collection - multiple rounds
    collected_objects = 0
    for i in range(15):  # Increased rounds
        collected = gc.collect()
        collected_objects += collected
        if collected == 0 and i > 5:
            break
        time.sleep(0.1)  # Small delay between collections

    print(f"   Collected {collected_objects} Python objects")

    if torch.cuda.is_available():
        before_memory = torch.cuda.memory_allocated() / (1024 ** 3)

        # Multiple strategies for GPU memory cleanup
        try:
            # Strategy 1: Empty cache multiple times with delays
            for _ in range(8):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(0.2)

            # Strategy 2: Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

            # Strategy 3: Force garbage collection on GPU
            if hasattr(torch.cuda, 'memory_reserved'):
                reserved_before = torch.cuda.memory_reserved() / (1024 ** 3)

            # Strategy 4: Try to free cached memory
            for _ in range(3):
                torch.cuda.empty_cache()
                time.sleep(0.5)

        except Exception as e:
            print(f"   Warning during GPU cleanup: {e}")

        after_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        freed_memory = before_memory - after_memory
        print(f"   Freed {freed_memory:.2f}GB GPU memory")

        # Extended recovery time based on amount of memory freed
        recovery_time = min(max(freed_memory * 0.5, 1.0), 5.0)
        print(f"   Recovery period: {recovery_time:.1f}s")
        time.sleep(recovery_time)


def progressive_cooldown(monitor, metrics, step_name="training"):
    """Progressive cooling system based on system state"""
    critical_issues, warning_issues, risk_level = monitor.adaptive_protection_check(metrics)

    if critical_issues or warning_issues:
        cooldown_time = monitor.get_recommended_cooldown()

        if critical_issues:
            print(f"CRITICAL THERMAL PROTECTION - {step_name.upper()}")
            for issue in critical_issues:
                print(f"   {issue}")
            monitor.forced_rest_count += 1
        elif warning_issues:
            print(f"Thermal warning detected - {step_name}")
            for issue in warning_issues:
                print(f"   {issue}")

        print(f"Progressive cooldown: {cooldown_time}s (Level {monitor.cooldown_level}, Risk: {risk_level})")

        # Enhanced cooling with memory cleanup
        ultra_aggressive_memory_cleanup()

        # Progressive cooling with status updates
        intervals = [0.25, 0.5, 0.75, 1.0]
        for i, interval in enumerate(intervals):
            sleep_time = cooldown_time * interval / len(intervals)
            time.sleep(sleep_time)

            # Check temperature during cooldown
            current_temp = monitor.get_gpu_temperature()
            if current_temp:
                print(f"   Cooldown {int((i + 1) * 25)}%: GPU {current_temp}°C")

        return True
    else:
        # Normal operation - quick cleanup and minimal rest
        ultra_aggressive_memory_cleanup()
        time.sleep(2)
        monitor.reset_consecutive_counters()
        return False


def find_all_checkpoints(save_dir):
    """Find all available checkpoints with detailed information"""
    if not os.path.exists(save_dir):
        print(f"Save directory {save_dir} does not exist")
        return []

    # Look for all checkpoint patterns
    checkpoint_patterns = [
        os.path.join(save_dir, "ep*.pth"),
        os.path.join(save_dir, "emergency_*.pth"),
        os.path.join(save_dir, "interrupted_*.pth"),
        os.path.join(save_dir, "auto_save_*.pth")
    ]

    all_checkpoints = []
    for pattern in checkpoint_patterns:
        all_checkpoints.extend(glob.glob(pattern))

    if not all_checkpoints:
        print(f"No checkpoints found in {save_dir}")
        return []

    # Sort by modification time (newest first)
    checkpoint_info = []
    for checkpoint in all_checkpoints:
        mtime = os.path.getmtime(checkpoint)
        basename = os.path.basename(checkpoint)

        # Extract epoch number
        epoch = 0
        if "ep" in basename and basename.startswith("ep"):
            try:
                epoch_str = basename.replace("ep", "").replace(".pth", "")
                epoch = int(epoch_str)
            except:
                pass
        elif "emergency_epoch_" in basename:
            try:
                epoch_str = basename.replace("emergency_epoch_", "").replace(".pth", "")
                epoch = int(epoch_str)
            except:
                pass
        elif "auto_save_epoch_" in basename:
            try:
                epoch_str = basename.replace("auto_save_epoch_", "").replace(".pth", "")
                epoch = int(epoch_str)
            except:
                pass

        checkpoint_info.append({
            'path': checkpoint,
            'epoch': epoch,
            'mtime': mtime,
            'basename': basename
        })

    # Sort by epoch number (highest first), then by modification time
    checkpoint_info.sort(key=lambda x: (x['epoch'], x['mtime']), reverse=True)

    return checkpoint_info


def find_latest_checkpoint(save_dir):
    """Find the latest checkpoint file with enhanced detection"""
    checkpoints = find_all_checkpoints(save_dir)

    if not checkpoints:
        return None, 0

    # Print all available checkpoints
    print(f"Available checkpoints in {save_dir}:")
    for i, cp in enumerate(checkpoints[:5]):  # Show top 5
        time_str = datetime.datetime.fromtimestamp(cp['mtime']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"   {i + 1}. {cp['basename']} (epoch {cp['epoch']}, {time_str})")

    latest = checkpoints[0]

    # Check checkpoint completion status
    try:
        checkpoint_data = torch.load(latest['path'], map_location='cpu')
        if isinstance(checkpoint_data, dict):
            training_completed = checkpoint_data.get('training_completed', False)
            next_epoch = checkpoint_data.get('next_epoch', latest['epoch'])

            if training_completed:
                print(f"Selected completed checkpoint: {latest['basename']}")
                print(f"Resume from epoch: {next_epoch}")
                return latest['path'], next_epoch
            else:
                print(f"Selected incomplete checkpoint: {latest['basename']}")
                print(f"Resume from epoch: {latest['epoch']}")
                return latest['path'], latest['epoch']
    except:
        pass

    print(f"Selected latest checkpoint: {latest['basename']}")
    print(f"Resume from epoch: {latest['epoch']}")

    return latest['path'], latest['epoch']


def load_checkpoint_robust(model, optimizer, checkpoint_path, device):
    """Enhanced checkpoint loading with comprehensive error handling"""
    try:
        print(f"Loading checkpoint: {checkpoint_path}")

        # Verify file exists and is readable
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file does not exist: {checkpoint_path}")
            return False

        file_size = os.path.getsize(checkpoint_path)
        if file_size < 1024:  # Less than 1KB is suspicious
            print(f"Checkpoint file seems corrupted (size: {file_size} bytes)")
            return False

        # Load checkpoint with error handling
        try:
            if device.type == 'cuda':
                checkpoint = torch.load(checkpoint_path, map_location=device)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            print(f"Failed to load checkpoint file: {e}")
            return False

        # Validate checkpoint structure
        if not isinstance(checkpoint, dict):
            print("Invalid checkpoint format")
            return False

        # Load model state
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("Model state loaded successfully")
            except Exception as e:
                print(f"Failed to load model state: {e}")
                return False
        else:
            print("No model_state_dict found in checkpoint")
            return False

        # Load optimizer state (optional)
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded successfully")
            except Exception as e:
                print(f"Could not load optimizer state: {e}")
                print("Will continue with fresh optimizer")

        # Get additional info
        epoch = checkpoint.get('epoch', 0)
        timestamp = checkpoint.get('timestamp', 'Unknown')
        training_params = checkpoint.get('training_params', {})

        print(f"Checkpoint timestamp: {timestamp}")
        print(f"Resuming from epoch: {epoch + 1}")

        if training_params:
            print(f"Previous training params: {training_params}")

        return True

    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return False


def auto_save_checkpoint(model, optimizer, epoch, save_dir, prefix="auto_save"):
    """Automatic checkpoint saving with error handling"""
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch + 1}.pth")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': datetime.datetime.now().isoformat(),
            'auto_save': True
        }

        # Save to temporary file first, then rename (atomic operation)
        temp_path = save_path + ".tmp"
        torch.save(checkpoint_data, temp_path)
        shutil.move(temp_path, save_path)

        print(f"Auto-saved: {os.path.basename(save_path)}")
        return True

    except Exception as e:
        print(f"Auto-save failed: {e}")
        return False


def ultra_safe_training_step(model_train, model, loss_history, eval_callback, optimizer, epoch,
                             epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda,
                             dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                             save_period, save_dir, local_rank, monitor, max_retries=3):
    """Ultra-safe training step with advanced thermal protection"""

    # Pre-training system assessment
    print(f"Pre-training system assessment for epoch {epoch + 1}...")
    metrics = monitor.get_comprehensive_metrics()
    print(f"   GPU: {metrics['gpu_percent']:.1f}% memory, {metrics['gpu_temp'] or 'N/A'}°C")
    print(f"   CPU: {metrics['cpu_percent']:.1f}% memory, {metrics['cpu_temp'] or 'N/A'}°C")

    # Progressive cooldown before training
    needs_cooldown = progressive_cooldown(monitor, metrics, "pre-training")

    for attempt in range(max_retries):
        try:
            print(f"Starting training step (attempt {attempt + 1}/{max_retries})")

            # Auto-save before training step
            auto_save_checkpoint(model, optimizer, epoch, save_dir, "pre_epoch")

            # Create a monitoring thread to check system during training
            training_interrupted = threading.Event()
            training_completed = threading.Event()  # Added: mark training completion

            def monitor_during_training():
                """Monitor system during training and set interrupt flag if needed"""
                while not training_interrupted.is_set() and not training_completed.is_set():
                    time.sleep(10)  # Check every 10 seconds

                    # If training is completed, stop monitoring
                    if training_completed.is_set():
                        break

                    current_metrics = monitor.get_comprehensive_metrics()
                    critical_issues, _, _ = monitor.adaptive_protection_check(current_metrics)

                    if critical_issues:
                        print("CRITICAL SYSTEM PROTECTION DURING TRAINING!")
                        for issue in critical_issues:
                            print(f"   {issue}")
                        training_interrupted.set()
                        break

            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_during_training, daemon=True)
            monitor_thread.start()

            # Perform training step with periodic checks
            try:
                fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                              epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda,
                              dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                              save_period, save_dir, local_rank)

                # Mark completion immediately after training
                training_completed.set()
                print("Training epoch completed successfully")

            finally:
                # Ensure monitoring thread is stopped
                training_completed.set()
                training_interrupted.set()
                monitor_thread.join(timeout=2)

            # Check if interrupted during training (not after completion)
            if training_interrupted.is_set() and not training_completed.is_set():
                raise RuntimeError("Training interrupted by thermal protection during execution")

            # Auto-save after successful training step
            auto_save_checkpoint(model, optimizer, epoch, save_dir, "post_epoch")

            # Post-training assessment and cooldown
            print("Post-training system assessment...")
            post_metrics = monitor.get_comprehensive_metrics()
            print(f"   GPU: {post_metrics['gpu_percent']:.1f}% memory, {post_metrics['gpu_temp'] or 'N/A'}°C")

            # Mandatory post-training cooldown
            progressive_cooldown(monitor, post_metrics, "post-training")

            print("Training step completed successfully")
            return True

        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM on attempt {attempt + 1}: {e}")
            ultra_aggressive_memory_cleanup()

            # Extended cooldown on OOM
            cooldown_time = 45 * (attempt + 1)  # Progressive increase
            print(f"OOM Recovery cooldown: {cooldown_time}s")
            time.sleep(cooldown_time)

            if attempt < max_retries - 1:
                print(f"Retrying after OOM recovery...")
                continue
            else:
                print("Failed after all OOM retry attempts")
                return False

        except Exception as e:
            print(f"Training error on attempt {attempt + 1}: {e}")

            # Emergency save on error
            try:
                auto_save_checkpoint(model, optimizer, epoch, save_dir, "error_recovery")
            except:
                pass

            # Get current system state
            error_metrics = monitor.get_comprehensive_metrics()
            critical_issues, _, _ = monitor.adaptive_protection_check(error_metrics)

            if critical_issues:
                print("Error may be thermal-related:")
                for issue in critical_issues:
                    print(f"   {issue}")

                # Extended thermal recovery
                cooldown_time = 90 * (attempt + 1)
                print(f"Thermal error recovery: {cooldown_time}s")
                ultra_aggressive_memory_cleanup()
                time.sleep(cooldown_time)
            else:
                # Regular error recovery
                ultra_aggressive_memory_cleanup()
                time.sleep(15 * (attempt + 1))

            if attempt < max_retries - 1:
                print(f"Retrying after error recovery...")
                continue
            else:
                print("Failed after all error retry attempts")
                return False

    return False


if __name__ == "__main__":
    # ============ Enhanced Configuration ============
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
    Freeze_batch_size = 20  # Keep original batch sizes
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 25  # Keep original batch sizes
    Freeze_Train = True

    # Keep original learning rates
    Init_lr = 3e-5
    Min_lr = Init_lr * 0.01
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 0.03
    lr_decay_type = 'cos'

    save_period = 1
    save_dir = 'logs_stu'
    eval_flag = True
    eval_period = 5

    VOCdevkit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "VOCdevkit"))
    dice_loss = True
    focal_loss = True
    cls_weights = np.array([1.0, 1.2, 1.5, 1.8], np.float32)
    num_workers = 0
    data_augmentation = False

    # ============ ADVANCED SYSTEM PROTECTION INITIALIZATION ============
    print("Advanced STU Training Protection System Initialized")
    print("=" * 80)

    # Initialize advanced system monitor
    monitor = AdvancedSystemMonitor()

    # Initial comprehensive system check
    print("Comprehensive Initial System Assessment:")
    initial_metrics = monitor.get_comprehensive_metrics()
    risk_level = monitor.predict_thermal_risk(initial_metrics)

    print(f"CPU: {initial_metrics['cpu_percent']:.1f}% memory, {initial_metrics['cpu_usage']:.1f}% usage")
    if initial_metrics['cpu_temp']:
        print(f"CPU Temperature: {initial_metrics['cpu_temp']:.1f}°C")

    print(f"GPU: {initial_metrics['gpu_percent']:.1f}% memory")
    if initial_metrics['gpu_temp']:
        print(f"GPU Temperature: {initial_metrics['gpu_temp']:.1f}°C")

    print(f"Thermal Risk Level: {risk_level}")

    # Emergency protection check
    critical_issues, warning_issues, _ = monitor.adaptive_protection_check(initial_metrics)
    if critical_issues:
        print("Critical system issues detected during startup:")
        for issue in critical_issues:
            print(f"   {issue}")
        print("System not ready for training. Please cool down and retry.")
        exit(1)
    elif warning_issues:
        print("System warnings detected:")
        for issue in warning_issues:
            print(f"   {issue}")
        print("Pre-training cooldown recommended...")
        progressive_cooldown(monitor, initial_metrics, "startup")

    # ============ RESUME TRAINING LOGIC ============
    latest_checkpoint, resume_epoch = find_latest_checkpoint(save_dir)

    if latest_checkpoint:
        print(f"RESUMING TRAINING from epoch {resume_epoch + 1}")
        print(f"Using checkpoint: {latest_checkpoint}")
        Init_Epoch = resume_epoch

        # Verify checkpoint integrity
        if os.path.exists(latest_checkpoint):
            file_size = os.path.getsize(latest_checkpoint) / (1024 * 1024)  # MB
            print(f"Checkpoint size: {file_size:.2f} MB")
        else:
            print("Checkpoint file not found!")
            exit(1)
    else:
        print("Starting fresh training (no checkpoints found)")
        resume_epoch = 0

    print("=" * 80)

    print("Starting Ultra-Safe STU Training with Advanced Protection")
    print(f"Batch sizes - Freeze: {Freeze_batch_size}, Unfreeze: {Unfreeze_batch_size}")
    print(f"Auto-save frequency: Multiple per epoch")
    print(f"Thermal protection: Advanced predictive system")

    # Initial cleanup before model creation
    ultra_aggressive_memory_cleanup()

    # Set random seed
    seed_everything(seed)

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

    print(f"Using device: {device}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

    # Model creation with thermal monitoring
    try:
        pre_model_metrics = monitor.get_comprehensive_metrics()
        print(f"Pre-model - GPU: {pre_model_metrics['gpu_percent']:.1f}%")

        model = create_stu_model(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone=backbone,
            input_shape=input_shape
        ).train()

        if model_path != '' and not latest_checkpoint:
            print(f"Loading pretrained weights: {model_path}")

        post_model_metrics = monitor.get_comprehensive_metrics()
        print(f"Post-model - GPU: {post_model_metrics['gpu_percent']:.1f}%")

        # Safety check after model creation with progressive cooldown
        progressive_cooldown(monitor, post_model_metrics, "post-model-creation")

    except Exception as e:
        print(f"Model creation failed: {e}")
        exit(1)

    print("Model created successfully")

    # Setup training components
    print("\nSetting up training components...")

    # Loss history
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # Mixed precision
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

    # Load datasets
    print("Loading datasets...")
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    print(f"Training samples: {num_train}, Validation samples: {num_val}")

    # Configuration display
    if local_rank == 0:
        print("\n" + "=" * 80)
        print(f"{'Parameter':<30} | {'Value':<40}")
        print("=" * 80)
        print(f"{'Model Type':<30} | {'STU (Ultra-Safe Mode)':<40}")
        print(f"{'Backbone':<30} | {backbone:<40}")
        print(f"{'Resume from Epoch':<30} | {Init_Epoch + 1:<40}")
        print(f"{'Batch Size (Freeze)':<30} | {Freeze_batch_size:<40}")
        print(f"{'Batch Size (Unfreeze)':<30} | {Unfreeze_batch_size:<40}")
        print(f"{'Learning Rate':<30} | {Init_lr:<40}")
        print(f"{'Thermal Protection':<30} | {'Advanced Predictive':<40}")
        if latest_checkpoint:
            print(f"{'Checkpoint':<30} | {os.path.basename(latest_checkpoint):<40}")
        print("=" * 80)

    # Training setup
    UnFreeze_flag = False

    # Determine if we should start with frozen or unfrozen backbone
    if Init_Epoch >= Freeze_Epoch:
        UnFreeze_flag = True
        print(f"Starting with unfrozen backbone (epoch {Init_Epoch + 1} >= {Freeze_Epoch})")
    elif Freeze_Train:
        model.freeze_backbone()
        print("Starting with frozen backbone")

    # Batch size selection
    batch_size = Unfreeze_batch_size if UnFreeze_flag else Freeze_batch_size

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

    # ============ ENHANCED CHECKPOINT LOADING ============
    if latest_checkpoint:
        print(f"\nLoading checkpoint from {latest_checkpoint}...")
        success = load_checkpoint_robust(model, optimizer, latest_checkpoint, device)
        if success:
            print("Successfully resumed from checkpoint")
        else:
            print("Failed to load checkpoint, starting fresh")
            Init_Epoch = 0

    # Calculate steps
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset too small for training")

    # Create datasets
    print("Creating datasets...")
    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path, data_augmentation)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path, data_augmentation)

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

    print("\nStarting Ultra-Safe Training Loop with Advanced Thermal Protection...")

    # ============ ULTRA-SAFE TRAINING LOOP ============
    training_start_time = time.time()

    try:
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            epoch_start_time = time.time()

            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{UnFreeze_Epoch}")
            if latest_checkpoint and epoch == Init_Epoch:
                print(f"Resumed from checkpoint: {os.path.basename(latest_checkpoint)}")

            # Display current thermal protection level
            print(f"Thermal Protection Level: {monitor.cooldown_level}/3")
            if monitor.forced_rest_count > 0:
                print(f"Forced rest sessions: {monitor.forced_rest_count}")

            print(f"{'=' * 60}")

            # Pre-epoch comprehensive system assessment
            metrics = monitor.get_comprehensive_metrics()
            risk_level = monitor.predict_thermal_risk(metrics)
            print(
                f"System Status - CPU: {metrics['cpu_percent']:.1f}% | GPU: {metrics['gpu_percent']:.1f}% | Risk: {risk_level}")
            if metrics['gpu_temp']:
                print(f"Temperatures - GPU: {metrics['gpu_temp']}°C", end="")
                if metrics['cpu_temp']:
                    print(f" | CPU: {metrics['cpu_temp']}°C")
                else:
                    print()

            # Check if we need pre-epoch rest
            critical_issues, warning_issues, _ = monitor.adaptive_protection_check(metrics)
            if critical_issues:
                print(f"Critical system protection triggered before epoch {epoch + 1}")
                for issue in critical_issues:
                    print(f"   {issue}")

                # Extended pre-epoch cooldown
                extended_cooldown = monitor.get_recommended_cooldown() + 60
                print(f"Extended pre-epoch cooldown: {extended_cooldown}s")
                ultra_aggressive_memory_cleanup()
                time.sleep(extended_cooldown)

                # Re-check after cooldown
                post_cooldown_metrics = monitor.get_comprehensive_metrics()
                post_critical, _, _ = monitor.adaptive_protection_check(post_cooldown_metrics)
                if post_critical:
                    print("System still critical after cooldown - stopping training")
                    break

            # Unfreeze handling with thermal protection
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                print(f"\nEpoch {epoch + 1}: Attempting to unfreeze backbone...")

                # Extra safety check before unfreezing
                unfreeze_metrics = monitor.get_comprehensive_metrics()
                unfreeze_critical, _, _ = monitor.adaptive_protection_check(unfreeze_metrics)
                if unfreeze_critical:
                    print("Cannot unfreeze - system not thermally stable")
                    print("Extended thermal recovery before unfreezing...")
                    ultra_aggressive_memory_cleanup()
                    time.sleep(180)  # 3-minute recovery

                    # Re-check
                    recheck_metrics = monitor.get_comprehensive_metrics()
                    recheck_critical, _, _ = monitor.adaptive_protection_check(recheck_metrics)
                    if recheck_critical:
                        print("Still not safe to unfreeze - skipping this epoch")
                        continue

                print("System stable - proceeding with unfreezing...")

                batch_size = Unfreeze_batch_size
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.unfreeze_backbone()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                # Recreate data loaders
                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate,
                                 sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate,
                                     sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

                # Extended stabilization after unfreezing
                print("Post-unfreezing stabilization...")
                ultra_aggressive_memory_cleanup()
                time.sleep(30)  # Extended stabilization time

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # Ultra-safe training step with advanced monitoring
            print(f"Starting ultra-safe training for epoch {epoch + 1}...")
            training_success = ultra_safe_training_step(
                model_train, model, loss_history, eval_callback, optimizer, epoch,
                epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda,
                dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                save_period, save_dir, local_rank, monitor
            )

            if not training_success:
                print(f"Training failed at epoch {epoch + 1}")

                # Enhanced emergency save with system diagnostics
                emergency_path = os.path.join(save_dir, f"emergency_epoch_{epoch + 1}.pth")
                try:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    failure_metrics = monitor.get_comprehensive_metrics()

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'timestamp': datetime.datetime.now().isoformat(),
                        'training_params': {
                            'batch_size': batch_size,
                            'unfreeze_flag': UnFreeze_flag,
                            'current_lr': optimizer.param_groups[0]['lr']
                        },
                        'failure_info': {
                            'failed_epoch': epoch + 1,
                            'system_metrics': failure_metrics,
                            'thermal_protection_level': monitor.cooldown_level,
                            'forced_rest_count': monitor.forced_rest_count,
                            'temperature_history': monitor.temperature_history[-5:],  # Last 5 readings
                            'consecutive_high_temp': monitor.consecutive_high_temp
                        }
                    }, emergency_path)
                    print(f"Enhanced emergency save with diagnostics: {emergency_path}")
                except Exception as e:
                    print(f"Emergency save failed: {e}")

                # Final system recovery attempt
                print("Final system recovery attempt...")
                ultra_aggressive_memory_cleanup()
                time.sleep(120)  # 2-minute recovery
                break

            # Enhanced checkpoint saving with system diagnostics
            save_path = os.path.join(save_dir, f"ep{epoch + 1:03d}.pth")
            try:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                current_metrics = monitor.get_comprehensive_metrics()

                checkpoint_data = {
                    'epoch': epoch,
                    'next_epoch': epoch + 1,  # Next epoch to train
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'timestamp': datetime.datetime.now().isoformat(),
                    'training_completed': True,  # Mark this epoch as completed
                    'training_params': {
                        'batch_size': batch_size,
                        'unfreeze_flag': UnFreeze_flag,
                        'current_lr': optimizer.param_groups[0]['lr']
                    },
                    'system_diagnostics': {
                        'system_metrics': current_metrics,
                        'thermal_protection_level': monitor.cooldown_level,
                        'forced_rest_count': monitor.forced_rest_count,
                        'temperature_history': monitor.temperature_history[-3:],  # Last 3 readings
                        'training_stable': monitor.consecutive_high_temp == 0
                    },
                    'training_config': {
                        'backbone': backbone,
                        'num_classes': num_classes,
                        'input_shape': input_shape,
                        'optimizer_type': optimizer_type,
                        'fp16': fp16
                    }
                }

                # Atomic save
                temp_path = save_path + ".tmp"
                torch.save(checkpoint_data, temp_path)
                shutil.move(temp_path, save_path)
                print(f"Enhanced checkpoint with diagnostics saved: {save_path}")

                # Keep only last 5 checkpoints to save space
                checkpoint_files = sorted(glob.glob(os.path.join(save_dir, "ep*.pth")))
                if len(checkpoint_files) > 5:
                    for old_checkpoint in checkpoint_files[:-5]:
                        try:
                            os.remove(old_checkpoint)
                            print(f"Removed old checkpoint: {os.path.basename(old_checkpoint)}")
                        except:
                            pass

            except Exception as e:
                print(f"Checkpoint save failed: {e}")

            # Mandatory inter-epoch cooling and system assessment
            print("Inter-epoch cooling and system assessment...")
            inter_epoch_metrics = monitor.get_comprehensive_metrics()

            # Progressive inter-epoch cooldown
            needs_extended_cooldown = progressive_cooldown(monitor, inter_epoch_metrics, "inter-epoch")

            # Additional cooldown based on epoch number and system stress
            if epoch > 0 and (epoch + 1) % 5 == 0:  # Every 5 epochs
                print(f"Extended maintenance cooldown after epoch {epoch + 1}")
                ultra_aggressive_memory_cleanup()
                time.sleep(15)  # Extended maintenance break

            # Show thermal trend
            if len(monitor.temperature_history) >= 2:
                recent_temps = [h['gpu_temp'] for h in monitor.temperature_history[-2:] if h['gpu_temp']]
                if len(recent_temps) >= 2:
                    temp_trend = recent_temps[-1] - recent_temps[0]
                    trend_symbol = "Rising" if temp_trend > 0 else "Falling" if temp_trend < 0 else "Stable"
                    print(f"Temperature trend: {trend_symbol} {temp_trend:+.1f}°C")

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

            # Reset consecutive high temperature counter if epoch completed successfully
            if training_success:
                monitor.reset_consecutive_counters()

            if distributed:
                dist.barrier()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        final_path = os.path.join(save_dir, "interrupted_final.pth")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        final_metrics = monitor.get_comprehensive_metrics()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': datetime.datetime.now().isoformat(),
            'interruption_reason': 'user_interrupt',
            'final_system_state': {
                'system_metrics': final_metrics,
                'thermal_protection_level': monitor.cooldown_level,
                'forced_rest_count': monitor.forced_rest_count,
                'temperature_history': monitor.temperature_history
            }
        }, final_path)
        print(f"Final save with system diagnostics: {final_path}")

    except Exception as e:
        print(f"\nFatal training error: {e}")
        import traceback

        traceback.print_exc()

        emergency_path = os.path.join(save_dir, "emergency_final.pth")
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            error_metrics = monitor.get_comprehensive_metrics()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'timestamp': datetime.datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'error_system_state': {
                    'system_metrics': error_metrics,
                    'thermal_protection_level': monitor.cooldown_level,
                    'forced_rest_count': monitor.forced_rest_count,
                    'temperature_history': monitor.temperature_history,
                    'consecutive_high_temp': monitor.consecutive_high_temp
                }
            }, emergency_path)
            print(f"Emergency final save with full diagnostics: {emergency_path}")
        except:
            print("Could not save emergency checkpoint")

    finally:
        # Ultra-comprehensive final cleanup
        print("\nUltra-comprehensive final cleanup...")
        if local_rank == 0 and loss_history:
            try:
                loss_history.writer.close()
            except:
                pass

        # Final system assessment
        final_metrics = monitor.get_comprehensive_metrics()
        print(f"Final System State:")
        print(f"   GPU: {final_metrics['gpu_percent']:.1f}% memory, {final_metrics['gpu_temp'] or 'N/A'}°C")
        print(f"   Thermal Protection Level: {monitor.cooldown_level}/3")
        print(f"   Total Forced Rest Sessions: {monitor.forced_rest_count}")

        ultra_aggressive_memory_cleanup()

        total_time = time.time() - training_start_time
        print(f"\nTraining session ended after {total_time / 3600:.2f} hours")
        print(f"Models saved in: {save_dir}")

        # Show final checkpoint info
        final_checkpoint, final_epoch = find_latest_checkpoint(save_dir)
        if final_checkpoint:
            print(f"Final checkpoint: {os.path.basename(final_checkpoint)} (epoch {final_epoch})")

        # Final thermal report
        if monitor.temperature_history:
            all_gpu_temps = [h['gpu_temp'] for h in monitor.temperature_history if h['gpu_temp']]
            if all_gpu_temps:
                max_temp = max(all_gpu_temps)
                avg_temp = sum(all_gpu_temps) / len(all_gpu_temps)
                print(f"Session Temperature Report: Max {max_temp:.1f}°C, Avg {avg_temp:.1f}°C")

    print("Ultra-Safe STU training with Advanced Thermal Protection completed!")