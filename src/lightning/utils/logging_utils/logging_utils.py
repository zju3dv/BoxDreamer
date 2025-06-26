import traceback
import wandb
import numpy as np
import torch.distributed as dist
from src.utils.log import INFO, ERROR
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from functools import wraps


class LoggingUtils:
    """Utility class for logging operations."""

    @staticmethod
    def log_nested_dict(logger, nested_dict, prefix=""):
        """Log a nested dictionary of values."""
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                LoggingUtils.log_nested_dict(logger, value, prefix=f"{prefix}{key}/")
            else:
                logger.log(f"{prefix}{key}", value)

    @staticmethod
    def log_metrics(loggers, metrics, step, prefix=""):
        """Log metrics to all provided loggers."""
        prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        for logger in loggers:
            logger.log_metrics(prefixed_metrics, step=step)

    @staticmethod
    def log_visualizations(loggers, vis_results, step, prefix="val_"):
        """Log visualization results to all loggers."""
        for k, figs in vis_results.items():
            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    if isinstance(figs[0], wandb.Plotly):
                        wandb.log({f"{prefix}{k}": figs}, step=step)
                    else:
                        logger.log_image(key=f"{prefix}{k}", images=figs, step=step)
                elif isinstance(logger, TensorBoardLogger):
                    try:
                        imgs = np.stack(figs).squeeze()
                        logger.experiment.add_images(
                            f"{prefix}{k}_{step}",
                            imgs,
                            global_step=step,
                            dataformats="NHWC",
                        )
                    except:
                        # Handle shape mismatch
                        imgs = figs
                        sub_imgs = {}
                        for idx, img in enumerate(imgs):
                            if img.shape not in sub_imgs:
                                sub_imgs[img.shape] = []
                            sub_imgs[img.shape].append(img)
                        idx = 0
                        for shape, imgs in sub_imgs.items():
                            imgs = np.stack(imgs)
                            logger.experiment.add_images(
                                f"{prefix}{k}_{step}_{idx}",
                                imgs,
                                global_step=step,
                                dataformats="NHWC",
                            )
                            idx += 1
                else:
                    raise NotImplementedError(f"Logger {logger} not implemented")

    @staticmethod
    def safe_operation(error_message="Operation failed", exit_on_error=False):
        """Decorator for safely executing operations with proper error
        handling.

        Usage:
            @LoggingUtils.safe_operation()
            def some_function():
                # code

            @LoggingUtils.safe_operation(error_message="Custom error", exit_on_error=True)
            def another_function():
                # code
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    ERROR(f"{error_message}: {e}")
                    tb = traceback.format_exc()
                    ERROR(f"Traceback: {tb}")
                    if exit_on_error:
                        exit(1)
                    # Try to ensure all processes pass the barrier
                    try:
                        dist.barrier()
                    except:
                        pass
                    return None

            return wrapper

        return decorator
