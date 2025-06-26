import os

import sys

from pytorch_lightning import LightningModule, Callback, Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase

import hydra
from omegaconf import DictConfig
from typing import List
from src.utils.customize.template_utils import *
from src.utils.log import *
import signal
import traceback
import cv2
import argparse
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

cv2.setNumThreads(0)
os.environ["HYDRA_FULL_ERROR"] = "1"
# os.environ["NCCL_P2P_DISABLE"] = "1"  #

trainer_instance = None
exp_name = None
use_hf = True
reproducibility = True
import warnings

warnings.filterwarnings("ignore")


def handle_cleanup(signum, frame):
    global trainer_instance
    global exp_name
    WARNING(f"Received signal {signum}. Initiating cleanup...")
    if trainer_instance is not None:
        try:
            cwd = hydra.utils.get_original_cwd()
            checkpoint_path = os.path.join(
                cwd, f"models/checkpoints/{exp_name}/error_checkpoint.ckpt"
            )
            trainer_instance.save_checkpoint(checkpoint_path)
            INFO(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            ERROR(f"Failed to save checkpoint: {e}")
        finally:
            pass
    INFO("Cleanup completed. Exiting now.")
    sys.exit(0)


def handle(config: DictConfig):
    global trainer_instance

    print_key_configs(config)

    if "seed" in config:
        seed_everything(config["seed"])

    # Init PyTorch Lightning model ⚡
    model: LightningModule = hydra.utils.instantiate(config["model"], _recursive_=False)

    # Init PyTorch Lightning datamodule ⚡
    datamodule = hydra.utils.instantiate(config["datamodule"])

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init PyTorch Lightning loggers ⚡
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for key in config["logger"]["in_use"]:
            logger.append(hydra.utils.instantiate(config["logger"][key]))
        if (
            "wandb" in config["logger"]["in_use"]
            and "save_dir" in config["logger"]["wandb"]
        ):
            os.makedirs(config["logger"]["wandb"]["save_dir"], exist_ok=True)

    # Init PyTorch Lightning trainer ⚡
    trainer: Trainer = hydra.utils.instantiate(
        # config["trainer"], callbacks=callbacks, logger=logger, plugins=DDPPlugin(find_unused_parameters=False)
        config["trainer"],
        callbacks=callbacks,
        logger=logger,
    )

    trainer_instance = trainer

    # Send some parameters from config to all lightning loggers
    log_hparams_to_all_loggers(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    resume_path = config.model.get("resume_ckpt", None)
    pretrain_path = config.model.get("pretrained_ckpt", None)

    # here‘s logic:
    # if resume_path is provided, we will resume training from the checkpoint
    # else if pretrain_path is provided, we will load the pre-trained model and train from scratch
    # Attention: if both resume_path and pretrain_path are provided, we will only resume training from the checkpoint
    # resume have higher priority than pretrain

    try:
        if config.mode == "train":
            resume_attempted = False  # Flag to track if resume was attempted

            # Attempt to resume training if resume_path and config.resume are provided
            if resume_path is not None and config.resume:
                resume_attempted = True
                if os.path.exists(resume_path):
                    INFO(f"Resuming training from checkpoint {resume_path}.")
                    trainer.fit(
                        model=model, datamodule=datamodule, ckpt_path=resume_path
                    )
                else:
                    WARNING(
                        f"Checkpoint not found at {resume_path}. Attempting to load pre-trained model if available."
                    )
                    resume_attempted = False  # Reset flag since resume failed

            # If resume was not successful or not attempted, try loading pre-trained model
            if not resume_attempted:
                pretrain_used = False  # Flag to track if pretrain was used

                if pretrain_path is not None and config.use_pretrained:
                    if os.path.exists(pretrain_path):
                        INFO(f"Loading pre-trained model from {pretrain_path}.")
                        model.load_pretrained_params(pretrain_path)
                        pretrain_used = True
                    else:
                        WARNING(
                            f"Pre-trained model not found at {pretrain_path}. Training from scratch."
                        )

                if pretrain_used or (
                    pretrain_path is None or not config.use_pretrained
                ):
                    if pretrain_used:
                        INFO("Starting training from pre-trained model.")
                    else:
                        INFO("Starting training from scratch.")
                    trainer.fit(model=model, datamodule=datamodule)

        elif config.mode == "test":
            if not use_hf:
                if pretrain_path is None:
                    raise ValueError(
                        "Please provide the path to the pre-trained model for testing."
                    )

                if not os.path.exists(pretrain_path):
                    raise FileNotFoundError(
                        f"Pre-trained model not found at {pretrain_path} for testing."
                    )

                INFO(f"Loading pre-trained model from {pretrain_path} for testing.")
                model.load_pretrained_params(pretrain_path)
                trainer.test(model=model, datamodule=datamodule)
            else:
                if reproducibility:
                    file = hf_hub_download(
                        "yyh929/BoxDreamer", "BoxDreamer-vitb-reproduce.safetensor"
                    )
                    INFO("Using reproducible checkpoint from huggingface.")
                else:
                    file = hf_hub_download(
                        "yyh929/BoxDreamer", "BoxDreamer-vitb.safetensor"
                    )
                    INFO("Using latest checkpoint from huggingface.")
                ckpt = load_file(file)
                model.load_state_dict(ckpt)
                trainer.test(model=model, datamodule=datamodule)

        else:
            raise ValueError(
                f"Invalid mode: {config.mode}. Valid modes are 'train' and 'test'."
            )

    except Exception as e:
        ERROR(
            f"An error occurred during training/testing: {e}\n{traceback.format_exc()}"
        )
        raise e

    # Make sure everything closed properly
    finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )


@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):
    signal.signal(signal.SIGINT, handle_cleanup)
    signal.signal(signal.SIGTERM, handle_cleanup)
    signal.signal(signal.SIGABRT, handle_cleanup)
    signal.signal(signal.SIGSEGV, handle_cleanup)
    global exp_name
    exp_name = config.exp_name
    try:
        handle(config)
    except Exception as e:
        ERROR(f"Unhandled exception: {e}", exc_info=True)
        # log traceback
        traceback.print_exc()
        if trainer_instance is not None:
            if isinstance(trainer_instance, LightningModule):
                try:
                    trainer_instance.save_checkpoint(
                        os.path.join(
                            hydra.utils.get_original_cwd(),
                            f"models/checkpoints/{exp_name}/error_checkpoint.ckpt",
                        )
                    )
                    INFO("Saved checkpoint before exiting due to error.")
                except Exception as save_e:
                    ERROR(f"Failed to save checkpoint: {save_e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--hf", action="store_true", help="Use huggingface safetensor", default=False
    )
    args.add_argument(
        "--reproducibility",
        action="store_true",
        help="Use reproducible checkpoint (from huggingface), only when hf is True",
        default=False,
    )
    args, unknown = args.parse_known_args()
    use_hf = args.hf
    sys.argv = sys.argv[:1] + unknown

    main()
