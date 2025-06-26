"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 22:51:11
Description:

"""
import torch
import math
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup scheduler wrapper that provides warmup functionality for any lr scheduler.

    Applies a learning rate warmup over the first few steps/epochs, then delegates to
    the base scheduler.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        base_scheduler=None,
        warmup_type="linear",
        last_step=-1,
    ):
        """
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of steps/epochs for warmup
            base_scheduler: The scheduler to use after warmup
            warmup_type: Type of warmup ('linear', 'exponential', or 'cosine')
            last_step: The index of the last step
        """
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.warmup_type = warmup_type
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_step = last_step

        # Initialize parent class
        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Still in warmup phase
            if self.warmup_type == "linear":
                # Linear warmup
                alpha = self.last_epoch / self.warmup_steps
                return [base_lr * alpha for base_lr in self.base_lrs]
            elif self.warmup_type == "exponential":
                # Exponential warmup
                alpha = self.last_epoch / self.warmup_steps
                return [base_lr * (alpha**3) for base_lr in self.base_lrs]
            elif self.warmup_type == "cosine":
                # Cosine warmup
                alpha = self.last_epoch / self.warmup_steps
                return [
                    base_lr * (1 - math.cos(alpha * math.pi / 2))
                    for base_lr in self.base_lrs
                ]
            else:
                raise ValueError(f"Warmup type {self.warmup_type} not supported")
        else:
            # Warmup is complete, use the base scheduler if available
            if self.base_scheduler:
                if hasattr(self.base_scheduler, "_last_lr"):
                    return self.base_scheduler._last_lr
                else:
                    # For some schedulers we need to call step() to update _last_lr
                    self.base_scheduler.step()
                    return self.base_scheduler._last_lr
            else:
                # If no base scheduler, keep the final warmup learning rate
                return [base_lr for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Also step the base scheduler if it exists and warmup is complete
        if self.base_scheduler and self.last_epoch >= self.warmup_steps:
            if hasattr(self.base_scheduler, "step") and callable(
                getattr(self.base_scheduler, "step")
            ):
                self.base_scheduler.step(epoch - self.warmup_steps)

        # Update the learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def build_optimizer(model, config):
    name = config["opt"]["optimizer"]
    lr = config["opt"]["true_lr"]

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=config["opt"]["adam_decay"]
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config["opt"]["adamw_decay"],
            amsgrad=config["opt"]["adamw_amsgrad"],
        )
    else:
        raise ValueError(f"opt.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer, max_step=None):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {"interval": config["opt"]["scheduler_invervel"]}
    name = config["opt"]["scheduler"]

    if config["opt"]["scheduler_invervel"] == "step":
        assert (
            max_step is not None
        ), "max_step should be provided when using step scheduler"
    else:
        max_step = None
        # For epoch-based scheduling, max_step is not required

    # Create the base scheduler
    base_scheduler = None
    if name == "MultiStepLR":
        base_scheduler = MultiStepLR(
            optimizer,
            config["opt"]["mslr_milestones"],
            gamma=config["opt"]["mslr_gamma"],
        )
    elif name == "CosineAnnealing":
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_step if max_step else config["opt"]["cosa_tmax"],
            eta_min=config["opt"]["cosa_eta_min"],
        )
    elif name == "ExponentialLR":
        base_scheduler = ExponentialLR(optimizer, config["opt"]["elr_gamma"])
    else:
        raise NotImplementedError(f"Scheduler {name} not implemented")

    # Check if warmup is enabled
    use_warmup = config["opt"].get("use_warmup", False)

    if use_warmup:
        # Determine warmup steps based on interval type
        if config["opt"]["scheduler_invervel"] == "step":
            # For step-based scheduling
            warmup_steps = config["opt"].get("warmup_steps", int(0.1 * max_step))
        else:
            # For epoch-based scheduling
            warmup_steps = config["opt"].get("warmup_epochs", 5)

        warmup_type = config["opt"].get("warmup_type", "linear")

        # Create the warmup scheduler wrapper
        final_scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            base_scheduler=base_scheduler,
            warmup_type=warmup_type,
        )

        scheduler.update({"scheduler": final_scheduler})
    else:
        # Use the base scheduler directly
        scheduler.update({"scheduler": base_scheduler})

    return scheduler
