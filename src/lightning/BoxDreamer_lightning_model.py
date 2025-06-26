import torch
import numpy as np
import copy
import gc
import pytorch_lightning as pl
import torch.distributed as dist
import hydra
from omegaconf import DictConfig

from src.models.BoxDreamerModel import BoxDreamer
from src.utils.log import INFO, ERROR
from .utils.vis.vis_utils import VisUtils
from .utils.metrics.metric_utils import Metrics
from .utils.optimizers.optimizers import build_optimizer, build_scheduler
from .utils.data_utils.data_utils import DataProcessor
from .utils.memory_utils.memory_utils import MemoryManager
from .utils.logging_utils.logging_utils import LoggingUtils
from torchmetrics.aggregation import MeanMetric


class PL_BoxDreamer(pl.LightningModule):
    """PyTorch Lightning module for BoxDreamer model.

    Handles training, validation, testing and logging.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the BoxDreamer model and related components."""
        super().__init__()

        self.save_hyperparameters()

        # Initialize model
        self.BoxDreamer = BoxDreamer(self.hparams)

        # Initialize loss functions
        train_loss_config = DictConfig(self.hparams["loss"]["train"])
        test_loss_config = DictConfig(self.hparams["loss"]["test"])
        self.train_loss = hydra.utils.instantiate(train_loss_config)
        self.test_loss = hydra.utils.instantiate(test_loss_config)

        # Initialize metrics and visualization handlers
        self.metrics_config = self.hparams["metrics"]
        self.vis_config = self.hparams["vis"]
        self.Vis_handler = VisUtils(self.vis_config)
        self.metrics_handler = Metrics(self.metrics_config)
        self.async_metrics_list = []

        # Initialize utility objects
        self.data_processor = DataProcessor()

        self.val_loss = MeanMetric()

    def load_pretrained_params(self, path):
        """Load pretrained model parameters from a checkpoint file.

        Args:
            path: Path to the checkpoint file
        """
        INFO(f"Loading pretrained weights from {path}")
        try:
            self.load_state_dict(torch.load(path)["state_dict"])
        except Exception as e:
            ERROR(f"Error loading pretrained weights: {e}")
            try:
                self.load_state_dict(torch.load(path)["state_dict"], strict=False)
                INFO("Loaded with strict=False")
            except Exception as e:
                # Some param mismatched, exclude them
                ERROR(f"Error loading pretrained weights with strict=False")
                # Parse the mismatched params from the error message
                import re

                pattern = re.compile(r"size mismatch for (.+):")
                all_match = pattern.findall(str(e))
                if all_match:
                    pretrained_param = torch.load(path)["state_dict"]
                    # Exclude the mismatched params
                    new_state_dict = pretrained_param.copy()
                    for k in pretrained_param.keys():
                        if k in all_match:
                            new_state_dict.pop(k)
                            INFO(f"Excluding {k} from pretrained weights")
                    self.load_state_dict(new_state_dict, strict=False)
                else:
                    raise e

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform a single training step.

        Args:
            batch: Input batch data
            batch_idx: Index of the current batch
            dataloader_idx: Index of the dataloader

        Returns:
            Loss value
        """
        self.BoxDreamer(batch)
        loss_value, loss_details = self.train_loss(batch)

        # Log metrics on main process at specified intervals
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            for logger in self.loggers:
                logger.log_metrics(
                    {"train_loss": loss_value.item()},
                    step=self.global_step,
                )

        # Log individual loss components
        for k, v in loss_details.items():
            self.log(f"train_{k}_loss", v, on_step=True, on_epoch=False, prog_bar=True)

        # Clean up memory
        MemoryManager.clean_variables(loss_details, batch)

        return loss_value

    def on_training_epoch_end(self):
        """Handle the end of a training epoch."""
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform a single validation step.

        Args:
            batch: Input batch data
            batch_idx: Index of the current batch
            dataloader_idx: Index of the dataloader
        """
        with torch.no_grad():
            self.BoxDreamer(batch)

            # Calculate loss
            loss_value, loss_details = self.test_loss(batch)
            loss_value = loss_value.item()

        # Log individual loss components
        for k, v in loss_details.items():
            self.log(f"val_{k}_loss", v, on_step=True, on_epoch=False, prog_bar=True)

        # Move batch data to CPU for metrics computation and visualization
        self.data_processor.back_to_cpu(batch)

        # Compute metrics and store for later aggregation
        batch_results = self.metrics_handler.compute_metrics(
            copy.deepcopy(batch), dataloader_id=dataloader_idx
        )
        self.val_loss.update(loss_value)

        # Add data for visualization
        self.Vis_handler.add_data(copy.deepcopy(batch), dataloader_id=dataloader_idx)
        self.Vis_handler.add_metrics(batch_results, dataloader_id=dataloader_idx)

        # Clean up memory
        MemoryManager.clean_variables(batch)

    @LoggingUtils.safe_operation(
        error_message="Error during val epoch end", exit_on_error=True
    )
    def on_validation_epoch_end(self):
        """Handle the end of a validation epoch."""
        # Gather metrics from all processes
        val_metrics = self.metrics_handler.get_metrics()
        dist.barrier()
        val_metrics = self.data_processor.gather_data(val_metrics, 0)

        self.log(
            "val_loss",
            self.val_loss.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Only process results on main process
        if self.trainer.global_rank == 0:
            # Aggregate metrics
            self.metrics_handler.set_metrics(val_metrics)
            agg_metrics = self.metrics_handler.aggregate_metrics()

            # Log metrics
            LoggingUtils.log_metrics(
                self.loggers,
                {"loss": self.val_loss.compute()},
                self.global_step,
                prefix="val_",
            )
            LoggingUtils.log_metrics(
                self.loggers, agg_metrics, self.global_step, prefix="val_"
            )

            # Generate and log visualizations
            self.Vis_handler.visualize_results()
            vis_results = self.Vis_handler.get_results()
            LoggingUtils.log_visualizations(
                self.loggers, vis_results, self.global_step, prefix="val_"
            )

            # Log metrics to Lightning's progress bar
            LoggingUtils.log_nested_dict(self, agg_metrics)

            dist.barrier()
        else:
            INFO(f"Rank {self.trainer.global_rank} waiting for barrier")
            dist.barrier()

        INFO(f"Rank {self.trainer.global_rank} finished")

        # Reset handlers and clear lists for next epoch
        self.Vis_handler.reset()
        self.metrics_handler.reset()
        self.val_loss.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform a single test step.

        Args:
            batch: Input batch data
            batch_idx: Index of the current batch
            dataloader_idx: Index of the dataloader
        """
        with torch.no_grad():
            self.BoxDreamer(batch)

        # Move batch data to CPU for metrics computation and visualization
        self.data_processor.back_to_cpu(batch)

        # Compute metrics
        batch_results = self.metrics_handler.compute_metrics(
            copy.deepcopy(batch), dataloader_id=dataloader_idx
        )

        # Add data for visualization
        self.Vis_handler.add_data(batch, dataloader_id=dataloader_idx)
        self.Vis_handler.add_metrics(batch_results, dataloader_id=dataloader_idx)

        # Clean up memory
        MemoryManager.clean_variables(batch)

    @LoggingUtils.safe_operation(
        error_message="Error during test epoch end", exit_on_error=True
    )
    def on_test_epoch_end(self):
        """Handle the end of a test epoch."""
        # Gather metrics from all processes
        test_metrics = self.metrics_handler.get_metrics()
        dist.barrier()
        test_metrics = self.data_processor.gather_data(test_metrics, 0)

        # Only process results on main process
        if self.trainer.global_rank == 0:
            # Aggregate metrics
            self.metrics_handler.set_metrics(test_metrics)
            agg_metrics = self.metrics_handler.aggregate_metrics()

            # Log metrics to Lightning's progress bar
            LoggingUtils.log_nested_dict(self, agg_metrics)

            # Save metrics to JSON file
            import pandas as pd

            with open("test_metrics.json", "w") as f:
                f.write(pd.Series(agg_metrics).to_json(indent=4))

            # Log metrics to all loggers
            LoggingUtils.log_metrics(
                self.loggers, agg_metrics, self.global_step, prefix="test_"
            )

            # Generate and log visualizations
            self.Vis_handler.visualize_results()
            vis_results = self.Vis_handler.get_results()
            LoggingUtils.log_visualizations(
                self.loggers, vis_results, self.global_step, prefix="test_"
            )

            # Clean up memory
            MemoryManager.clean_variables(vis_results, test_metrics, agg_metrics)

        # Reset handlers for next test
        self.Vis_handler.reset()
        self.metrics_handler.reset()

        gc.collect()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns:
            List of optimizers and schedulers
        """
        optimizer = build_optimizer(self, self.hparams)

        # Get the number of training steps
        num_steps = self.trainer.estimated_stepping_batches
        INFO(f"num_steps: {num_steps}")
        # TODO: check if this is correct
        # if max_steps is None or inf
        if num_steps is None or num_steps == float("inf"):
            num_steps = None

        scheduler = build_scheduler(self.hparams, optimizer, max_step=num_steps)
        return [optimizer], [scheduler]

    def reset_optimizers(self):
        """Reset optimizers and schedulers to their initial state."""
        for idx, _ in enumerate(self.optimizers()):
            optimizer = build_optimizer(self, self.hparams)
            self.optimizers()[idx] = optimizer

        for idx in range(len(self.lr_schedulers())):
            scheduler = build_scheduler(self.hparams, self.optimizers()[idx])
            self.lr_schedulers()[idx] = scheduler
