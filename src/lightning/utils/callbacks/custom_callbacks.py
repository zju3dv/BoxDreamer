from pytorch_lightning import Callback
import pytorch_lightning as pl
import os
import torch
from src.utils.log import *
from typing import Optional, List, Union
import torchvision.transforms as T
from matplotlib import pyplot as plt

class ExampleCallback(Callback):
    def __init__(self):
        pass

    def on_init_start(self, trainer):
        print("Starting to initialize trainer!")

    def on_init_end(self, trainer):
        print("Trainer is initialized now.")

    def on_train_end(self, trainer, pl_module):
        print("Do something when training ends.")


class UnfreezeModelCallback(Callback):
    """
    Unfreeze all model parameters after a few epochs.
    """

    def __init__(self, wait_epochs=5):
        self.wait_epochs = wait_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.wait_epochs:
            for param in pl_module.model.model.parameters():
                param.requires_grad = True


class NetworkMonitor(Callback):
    def __init__(self, 
                 visualize_every_n_batches=10,
                 visualize_types=('feature_map', 'attention'),
                 log_to_loggers=True,
                 save_dir='./visualizations',
                 names=None):
                 
        super().__init__()
        self.visualize_every_n_batches = visualize_every_n_batches
        self.visualize_types = visualize_types
        self.log_to_loggers = log_to_loggers
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.hooks = []
        self.names = names

    def on_train_start(self, trainer, pl_module):
        if self.names is None:
            INFO('No module names provided, all modules will be visualized')
        self.trainer = trainer
        for name, module in pl_module.named_modules():
            # INFO(f'Checking module {name}')
            if self.names is None or name in self.names:
                INFO(f'Adding hook for {name}')
                hook = module.register_forward_hook(self.save_activation(name))
                self.hooks.append(hook)

    def save_activation(self, name):
        def hook(module, input, output):
            if 'feature_map' in self.visualize_types:
                self.visualize_feature_map(output, name)
            if 'attention' in self.visualize_types:
                self.visualize_feature_map(output, name)
        return hook
    
    def visualize_feature_map(self, feature_map, name):
        # Use average pooling or select one channel for visualization
        # INFO(f"name: {name}, feature map type: {type(feature_map)}")
        
        try:
            if isinstance(feature_map, torch.Tensor) and feature_map.size(0) != 0:
                if feature_map.dim() == 4:  # Batch, Channels, Height, Width
                    # INFO(f'Visualizing feature map for {name}; tensor shape: {feature_map.shape}')
                    feature_map = feature_map[0]  # Select first sample
                    feature_map = torch.mean(feature_map, dim=0)  # Average over channels
                    self._save_and_log_image(feature_map, self.trainer.global_step, f'feature_map_{name}')
                elif feature_map.dim() == 3:  # Batch, Seq, Channels
                    # INFO(f'Visualizing attention for {name}; tensor shape: {feature_map.shape}')
                    feature_map = feature_map[0]
                
                    # unpatchify the tokenized image to (sqrt(n), sqrt(n), c)
                    N = int(feature_map.size(0)**0.5)
                    C = feature_map.size(1)
                    
                    feature_map = feature_map.transpose(0, 1).reshape(C, N, N)
                    
                    feature_map = torch.mean(feature_map, dim=0)  # Average over channels
                    self._save_and_log_image(feature_map, self.trainer.global_step, f'feature_map_{name}')
                    
                    
        except Exception as e:
            pass

    def _save_and_log_image(self, tensor: torch.Tensor, global_step: int, content_type: str, save_rgb: bool = False):
        if (global_step + 1) % self.visualize_every_n_batches == 0:
            try:
                if save_rgb:
                    tensor = tensor
                    tensor = T.ToPILImage()(tensor)
                    filename = f"{self.save_dir}/{content_type}_step{global_step}.png"
                    tensor.save(filename)
                else:
                    tensor = tensor.detach().cpu()
                    plt.figure(figsize=(6, 6))
                    plt.imshow(tensor, cmap='viridis')  # Use a common colormap
                    plt.colorbar()
                    plt.title(content_type)
                    plt.axis('off')

                    filename = f"{self.save_dir}/{content_type}_step{global_step}.png"
                    plt.savefig(filename, bbox_inches='tight')
                    plt.close()

                if self.log_to_loggers:
                    for logger in self.trainer.loggers:
                        logger.log_image(key=content_type, images=[filename], step=global_step)
                        
            except Exception as e:
                print(f"Error saving or logging image: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (self.trainer.global_step + 1) % self.visualize_every_n_batches == 0:
            INFO(f'Visualizing at step {self.trainer.global_step}')
            # save query image
            self.global_step = trainer.global_step
            self.trainer = trainer
            bs = batch['query']['img'].size(0)
            for i in range(bs):
                img = batch['query']['img'][i]
                self._save_and_log_image(img, self.global_step, f'query_image_{i}', save_rgb=True)

    def on_train_end(self, trainer, pl_module):
        for hook in self.hooks:
            hook.remove()