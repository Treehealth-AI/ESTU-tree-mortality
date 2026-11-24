import os
import torch
import gc
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm
from utils.sobel_loss import CombinedLoss
from sklearn.metrics import f1_score
import numpy as np
import csv
import datetime


def extract_edges_simple(masks):
    """
    Simple edge extraction function for edge supervision during training
    """
    # Sobel operators
    sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)

    if masks.is_cuda:
        sobel_x = sobel_x.cuda()
        sobel_y = sobel_y.cuda()

    # Convert masks to float and add channel dimension
    if masks.dim() == 3:
        masks_float = masks.unsqueeze(1).float()
    else:
        masks_float = masks.float()

    # Apply Sobel operators
    grad_x = torch.nn.functional.conv2d(masks_float, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(masks_float, sobel_y, padding=1)

    # Calculate edge strength
    edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    edges = (edges > 0.1).float()  # Binarization

    return edges.squeeze(1)  # Return [B, H, W]


def get_lr(optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def clear_memory():
    """Enhanced memory clearing function - silent version"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    # Removed all print statements for silent operation


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                  epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss,
                  cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    """
    Training function for Sobel-enhanced U-Net with periodic F1 score calculation and detailed logging
    Now includes enhanced memory management for stable training with larger batch sizes
    """
    total_loss = 0
    total_seg_loss = 0
    total_edge_loss = 0
    val_loss = 0

    # Memory management variables
    memory_clear_interval_train = 50
    memory_clear_interval_val = 30
    deep_clear_interval = 200

    # Determine if this epoch should calculate F1 score and detailed logging
    calculate_f1_this_epoch = (epoch + 1) % 5 == 0

    # Setup detailed metrics logging with separate CSV files
    if local_rank == 0 and calculate_f1_this_epoch:
        # Create CSV file paths for different metric categories
        log_dir = loss_history.log_dir if hasattr(loss_history, 'log_dir') else save_dir

        loss_csv_path = os.path.join(log_dir, 'loss_metrics.csv')
        f1_csv_path = os.path.join(log_dir, 'f1_metrics.csv')
        eval_csv_path = os.path.join(log_dir, 'evaluation_metrics.csv')
        training_csv_path = os.path.join(log_dir, 'training_info.csv')

        # Initialize CSV files with headers if they don't exist
        if not os.path.exists(loss_csv_path):
            with open(loss_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'total_loss', 'seg_loss', 'edge_loss', 'val_loss'])

        if not os.path.exists(f1_csv_path):
            with open(f1_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Updated header to include both training and validation F1
                writer.writerow(['timestamp', 'epoch', 'train_f1_score', 'val_f1_score'])

        if not os.path.exists(eval_csv_path):
            with open(eval_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'miou', 'mpa', 'accuracy'])

        if not os.path.exists(training_csv_path):
            with open(training_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'learning_rate', 'training_speed_it_per_sec'])
    else:
        loss_csv_path = f1_csv_path = eval_csv_path = training_csv_path = None

    # Create loss function (create only once to avoid repeated creation)
    if dice_loss or focal_loss:
        loss_fn = CombinedLoss(
            num_classes=num_classes,
            dice_loss=dice_loss,
            focal_loss=focal_loss,
            cls_weights=cls_weights,  # Pass original numpy array
            edge_weight=0.3
        )
        if cuda:
            loss_fn = loss_fn.cuda(local_rank)

    if local_rank == 0:
        print('Start Train')
        epoch_desc = f'Epoch {epoch + 1}/{Epoch}'
        if calculate_f1_this_epoch:
            epoch_desc += ' [F1 Tracking]'
        pbar = tqdm(total=epoch_step, desc=epoch_desc, postfix=dict, mininterval=0.3)

    # Training phase
    model_train.train()
    training_start_time = datetime.datetime.now()

    # Initialize training F1 tracking with memory-efficient approach
    if calculate_f1_this_epoch:
        # Use sampling strategy to reduce memory usage
        train_preds = []
        train_targets = []
        f1_sample_rate = max(1, epoch_step // 200)  # Sample every N iterations
    else:
        train_preds = None
        train_targets = None


    clear_memory()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                images = images.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()

        if not fp16:
            # Forward pass
            outputs = model_train(images)

            # Check output type
            if isinstance(outputs, tuple) and len(outputs) == 2:
                # Sobel-enhanced model: dual output
                seg_outputs, edge_outputs = outputs

                # Use combined loss function
                if dice_loss or focal_loss:
                    main_loss = loss_fn((seg_outputs, edge_outputs), pngs)
                else:
                    main_loss = torch.nn.functional.cross_entropy(seg_outputs, pngs.long(), weight=weights)

                # Calculate separate losses for recording
                seg_loss = torch.nn.functional.cross_entropy(seg_outputs, pngs.long(), weight=weights)
                edge_target = extract_edges_simple(pngs)
                edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    edge_outputs.squeeze(1), edge_target.float()
                )

                total_seg_loss += seg_loss.item()
                total_edge_loss += edge_loss.item()

                # Memory-efficient F1 collection with sampling
                if calculate_f1_this_epoch and iteration % f1_sample_rate == 0:
                    with torch.no_grad():
                        preds = torch.argmax(seg_outputs, dim=1)
                        # Sample pixels to reduce memory usage
                        sample_size = min(1000, preds.numel())
                        indices = torch.randperm(preds.numel())[:sample_size]
                        preds_flat = preds.flatten()
                        pngs_flat = pngs.flatten()

                        train_preds.extend(preds_flat[indices].cpu().numpy())
                        train_targets.extend(pngs_flat[indices].cpu().numpy())

                        # Clear variables immediately
                        del preds, preds_flat, pngs_flat, indices
            else:
                # Standard model: single output
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels, weights, num_classes)
                    main_loss = main_dice
                else:
                    main_loss = torch.tensor(0.0).cuda()

                if focal_loss:
                    main_focal = Focal_Loss(outputs, pngs, weights, num_classes, alpha=0.5, gamma=2)
                    main_loss = main_loss + main_focal

                if not dice_loss and not focal_loss:
                    main_ce = CE_Loss(outputs, pngs, weights, num_classes)
                    main_loss = main_ce

                # Memory-efficient F1 collection with sampling
                if calculate_f1_this_epoch and iteration % f1_sample_rate == 0:
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        # Sample pixels to reduce memory usage
                        sample_size = min(1000, preds.numel())
                        indices = torch.randperm(preds.numel())[:sample_size]
                        preds_flat = preds.flatten()
                        pngs_flat = pngs.flatten()

                        train_preds.extend(preds_flat[indices].cpu().numpy())
                        train_targets.extend(pngs_flat[indices].cpu().numpy())

                        # Clear variables immediately
                        del preds, preds_flat, pngs_flat, indices

            main_loss.backward()
            optimizer.step()
        else:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model_train(images)

                if isinstance(outputs, tuple) and len(outputs) == 2:
                    # Sobel-enhanced model
                    seg_outputs, edge_outputs = outputs

                    if dice_loss or focal_loss:
                        main_loss = loss_fn((seg_outputs, edge_outputs), pngs)
                    else:
                        main_loss = torch.nn.functional.cross_entropy(seg_outputs, pngs.long(), weight=weights)

                    seg_loss = torch.nn.functional.cross_entropy(seg_outputs, pngs.long(), weight=weights)
                    edge_target = extract_edges_simple(pngs)
                    edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        edge_outputs.squeeze(1), edge_target.float()
                    )

                    total_seg_loss += seg_loss.item()
                    total_edge_loss += edge_loss.item()

                    # Memory-efficient F1 collection with sampling
                    if calculate_f1_this_epoch and iteration % f1_sample_rate == 0:
                        with torch.no_grad():
                            preds = torch.argmax(seg_outputs, dim=1)
                            # Sample pixels to reduce memory usage
                            sample_size = min(1000, preds.numel())
                            indices = torch.randperm(preds.numel())[:sample_size]
                            preds_flat = preds.flatten()
                            pngs_flat = pngs.flatten()

                            train_preds.extend(preds_flat[indices].cpu().numpy())
                            train_targets.extend(pngs_flat[indices].cpu().numpy())

                            # Clear variables immediately
                            del preds, preds_flat, pngs_flat, indices
                else:
                    # Standard model
                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels, weights, num_classes)
                        main_loss = main_dice
                    else:
                        main_loss = torch.tensor(0.0).cuda()

                    if focal_loss:
                        main_focal = Focal_Loss(outputs, pngs, weights, num_classes, alpha=0.5, gamma=2)
                        main_loss = main_loss + main_focal

                    if not dice_loss and not focal_loss:
                        main_ce = CE_Loss(outputs, pngs, weights, num_classes)
                        main_loss = main_ce

                    # Memory-efficient F1 collection with sampling
                    if calculate_f1_this_epoch and iteration % f1_sample_rate == 0:
                        with torch.no_grad():
                            preds = torch.argmax(outputs, dim=1)
                            # Sample pixels to reduce memory usage
                            sample_size = min(1000, preds.numel())
                            indices = torch.randperm(preds.numel())[:sample_size]
                            preds_flat = preds.flatten()
                            pngs_flat = pngs.flatten()

                            train_preds.extend(preds_flat[indices].cpu().numpy())
                            train_targets.extend(pngs_flat[indices].cpu().numpy())

                            # Clear variables immediately
                            del preds, preds_flat, pngs_flat, indices

            scaler.scale(main_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += main_loss.item()


        if (iteration + 1) % memory_clear_interval_train == 0:
            torch.cuda.empty_cache()


        if (iteration + 1) % deep_clear_interval == 0:
            clear_memory()


        if local_rank == 0:
            postfix_dict = {
                'total_loss': total_loss / (iteration + 1),
                'lr': get_lr(optimizer)
            }
            if total_seg_loss > 0:
                postfix_dict['seg_loss'] = total_seg_loss / (iteration + 1)
                postfix_dict['edge_loss'] = total_edge_loss / (iteration + 1)

            pbar.set_postfix(**postfix_dict)
            pbar.update(1)

    # Calculate training speed
    training_end_time = datetime.datetime.now()
    training_duration = (training_end_time - training_start_time).total_seconds()
    training_speed = epoch_step / training_duration if training_duration > 0 else 0

    # Calculate training F1 score with error handling
    train_f1 = None
    if calculate_f1_this_epoch and train_preds:
        try:
            train_f1 = f1_score(train_targets, train_preds, average='weighted', zero_division=0)
        except Exception as e:
            print(f"Error calculating training F1 score: {e}")
            train_f1 = 0.0
        finally:
            # Clear F1 calculation data
            del train_preds, train_targets
            clear_memory()

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        if train_f1 is not None:
            print(f'Training F1 Score: {train_f1:.4f}')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # Validation phase with conditional F1 calculation and memory management
    model_train.eval()

    # Initialize validation F1 tracking with memory-efficient approach
    if calculate_f1_this_epoch:
        val_preds = []
        val_targets = []
        val_f1_sample_rate = max(1, epoch_step_val // 100)  # Sample every N iterations
    else:
        val_preds = None
        val_targets = None


    clear_memory()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        images, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                images = images.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs = model_train(images)

            if isinstance(outputs, tuple) and len(outputs) == 2:
                # Sobel-enhanced model
                seg_outputs, edge_outputs = outputs

                if dice_loss or focal_loss:
                    loss = loss_fn((seg_outputs, edge_outputs), pngs)
                else:
                    loss = torch.nn.functional.cross_entropy(seg_outputs, pngs.long(), weight=weights)

                # Memory-efficient F1 collection with sampling
                if calculate_f1_this_epoch and iteration % val_f1_sample_rate == 0:
                    preds = torch.argmax(seg_outputs, dim=1)
                    # Sample pixels to reduce memory usage
                    sample_size = min(500, preds.numel())  # Smaller sample for validation
                    indices = torch.randperm(preds.numel())[:sample_size]
                    preds_flat = preds.flatten()
                    pngs_flat = pngs.flatten()

                    val_preds.extend(preds_flat[indices].cpu().numpy())
                    val_targets.extend(pngs_flat[indices].cpu().numpy())

                    # Clear variables immediately
                    del preds, preds_flat, pngs_flat, indices

            else:
                # Standard model
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels, weights, num_classes)
                    loss = main_dice
                else:
                    loss = torch.tensor(0.0).cuda()

                if focal_loss:
                    main_focal = Focal_Loss(outputs, pngs, weights, num_classes, alpha=0.5, gamma=2)
                    loss = loss + main_focal

                if not dice_loss and not focal_loss:
                    main_ce = CE_Loss(outputs, pngs, weights, num_classes)
                    loss = main_ce

                # Memory-efficient F1 collection with sampling
                if calculate_f1_this_epoch and iteration % val_f1_sample_rate == 0:
                    preds = torch.argmax(outputs, dim=1)
                    # Sample pixels to reduce memory usage
                    sample_size = min(500, preds.numel())  # Smaller sample for validation
                    indices = torch.randperm(preds.numel())[:sample_size]
                    preds_flat = preds.flatten()
                    pngs_flat = pngs.flatten()

                    val_preds.extend(preds_flat[indices].cpu().numpy())
                    val_targets.extend(pngs_flat[indices].cpu().numpy())

                    # Clear variables immediately
                    del preds, preds_flat, pngs_flat, indices

            val_loss += loss.item()


            if (iteration + 1) % memory_clear_interval_val == 0:
                torch.cuda.empty_cache()


            if (iteration + 1) % 100 == 0:
                clear_memory()

            if local_rank == 0:
                pbar.set_postfix(**{
                    'val_loss': val_loss / (iteration + 1)
                })
                pbar.update(1)

    # Calculate validation F1 score only for designated epochs
    val_f1 = None
    if local_rank == 0:
        if calculate_f1_this_epoch and val_preds:
            try:
                val_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)
            except Exception as e:
                print(f"Error calculating validation F1 score: {e}")
                val_f1 = 0.0
            finally:
                # Clear F1 calculation data
                del val_preds, val_targets
                clear_memory()

            # Update progress bar with F1 scores
            postfix_dict = {
                'val_loss': val_loss / epoch_step_val,
                'val_f1': val_f1
            }
            if train_f1 is not None:
                postfix_dict['train_f1'] = train_f1
            pbar.set_postfix(**postfix_dict)

        pbar.close()
        print('Finish Validation')
        if val_f1 is not None:
            print(f'Validation F1 Score: {val_f1:.4f}')

        # Record loss history
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)

        # Record additional losses and F1 scores
        if hasattr(loss_history, 'writer'):
            if total_seg_loss > 0:
                loss_history.writer.add_scalar('Loss/Train_Seg', total_seg_loss / epoch_step, epoch)
                loss_history.writer.add_scalar('Loss/Train_Edge', total_edge_loss / epoch_step, epoch)
                loss_history.writer.add_scalar('Training/Speed_it_per_sec', training_speed, epoch)
            if train_f1 is not None:
                loss_history.writer.add_scalar('F1_Score/Training', train_f1, epoch)
            if val_f1 is not None:
                loss_history.writer.add_scalar('F1_Score/Validation', val_f1, epoch)

        # Run evaluation callback and capture metrics
        miou, mpa, accuracy = None, None, None
        if calculate_f1_this_epoch:
            # Capture output from eval callback
            import sys
            from io import StringIO

            # Capture stdout to extract evaluation metrics
            old_stdout = sys.stdout
            captured_output = StringIO()
            sys.stdout = captured_output

            try:
                eval_callback.on_epoch_end(epoch + 1, model_train)
                output = captured_output.getvalue()

                # Parse mIoU, mPA, and Accuracy from output
                lines = output.split('\n')
                for line in lines:
                    if 'mIoU:' in line and 'mPA:' in line and 'Accuracy:' in line:
                        # Parse line like "===> mIoU: 37.0; mPA: 65.99; Accuracy: 66.63"
                        parts = line.split(';')
                        for part in parts:
                            if 'mIoU:' in part:
                                miou = float(part.split('mIoU:')[1].strip())
                            elif 'mPA:' in part:
                                mpa = float(part.split('mPA:')[1].strip())
                            elif 'Accuracy:' in part:
                                accuracy = float(part.split('Accuracy:')[1].strip())
                        break
            except Exception as e:
                print(f"Error capturing evaluation metrics: {e}")
            finally:
                sys.stdout = old_stdout
                # Print the captured output
                print(captured_output.getvalue(), end='')
        else:
            eval_callback.on_epoch_end(epoch + 1, model_train)


        if calculate_f1_this_epoch and loss_csv_path:
            current_lr = get_lr(optimizer)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Write loss metrics
            with open(loss_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    epoch + 1,
                    f"{total_loss / epoch_step:.6f}",
                    f"{total_seg_loss / epoch_step:.6f}" if total_seg_loss > 0 else "N/A",
                    f"{total_edge_loss / epoch_step:.6f}" if total_edge_loss > 0 else "N/A",
                    f"{val_loss / epoch_step_val:.6f}"
                ])

            # Write F1 metrics (both training and validation)
            with open(f1_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    epoch + 1,
                    f"{train_f1:.6f}" if train_f1 is not None else "N/A",
                    f"{val_f1:.6f}" if val_f1 is not None else "N/A"
                ])

            # Write evaluation metrics
            if miou is not None:
                with open(eval_csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp,
                        epoch + 1,
                        f"{miou:.2f}",
                        f"{mpa:.2f}",
                        f"{accuracy:.2f}"
                    ])

            # Write training info
            with open(training_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    epoch + 1,
                    f"{current_lr:.2e}",
                    f"{training_speed:.2f}"
                ])


            if miou is not None:
                print(f"Metrics saved - mIoU: {miou:.2f}, mPA: {mpa:.2f}, Accuracy: {accuracy:.2f}")

        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        if train_f1 is not None and val_f1 is not None:
            print('Total Loss: %.3f || Val Loss: %.3f || Train F1: %.4f || Val F1: %.4f' % (
                total_loss / epoch_step, val_loss / epoch_step_val, train_f1, val_f1))
        elif val_f1 is not None:
            print('Total Loss: %.3f || Val Loss: %.3f || Val F1: %.4f' % (
                total_loss / epoch_step, val_loss / epoch_step_val, val_f1))
        else:
            print('Total Loss: %.3f || Val Loss: %.3f' % (
                total_loss / epoch_step, val_loss / epoch_step_val))

        if total_seg_loss > 0:
            print('Seg Loss: %.3f || Edge Loss: %.3f' % (total_seg_loss / epoch_step, total_edge_loss / epoch_step))

        # Save model
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            if train_f1 is not None and val_f1 is not None:
                torch.save(model.state_dict(),
                           os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f-train_f1_%.4f-val_f1_%.4f.pth" % (
                               epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val, train_f1, val_f1)))
            elif val_f1 is not None:
                torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f-val_f1_%.4f.pth" % (
                    epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val, val_f1)))
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
                    epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))


    clear_memory()