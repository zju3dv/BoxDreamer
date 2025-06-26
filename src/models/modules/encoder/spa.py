from .base import PretrainedModelWrapper
from torchvision import models
import torch
import os
import sys

# 3rd party imports
# 3rd dir is under repo root, so we need to add it to sys.path
HERE_PATH = os.path.normpath(os.path.dirname(__file__))
SPA_REPO_PATH = os.path.normpath(os.path.join(HERE_PATH, '../../../../three'))
SPA_LIB_PATH = os.path.join(SPA_REPO_PATH, 'spa')
# check the presence of models directory in repo to be sure its cloned

if os.path.isdir(SPA_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, SPA_LIB_PATH)
else:
    raise ImportError(f"spa is not initialized, could not find: {SPA_LIB_PATH}.\n "
                      "Did you forget to run 'git submodule update --init --recursive' ?")

from spa.models import spa_vit_base_patch16, spa_vit_large_patch16

class SPAModelWrapper(PretrainedModelWrapper):
    def __init__(self, ckpt_path: None, cfg):
        super().__init__(model_name_or_path='spa')

        self.ckpt_path = ckpt_path
        self.model_type = cfg.get('model_type', 'spa_vit_base_patch16')
        assert self.model_type in ['spa_vit_base_patch16', 'spa_vit_large_patch16']
        self.model = None
        self.device = None
        self.load_model()
        
    def get_device(self):
        return self.device
    
    def to_device(self, device):
        self.model = self.model.to(device)
        self.device = device
        
    def load_model(self, device='cuda'):
        self.device = device
        if self.model_type == 'spa_vit_base_patch16':
            self.model = spa_vit_base_patch16(pretrained=True).to(device)
        elif self.model_type == 'spa_vit_large_patch16':
            self.model = spa_vit_large_patch16(pretrained=True).to(device)
        else:
            raise ValueError(f'Unknown model type: {self.model_type}')
        
        self.model.eval()
        self.model.freeze()
        
    def predict(self, input_tensor):
        if input_tensor.dim() == 5:
            # BTCHW
            B, T, C, H, W = input_tensor.size()
            seq_feats = []
            for i in range(B):
                seq_feats.append(self.predict(input_tensor[i]))
            # stack to original shape
            return torch.stack(seq_feats, dim=0)
        else:
            B, C, H, W = input_tensor.size()
            
            with torch.no_grad():
                ret = self.model(input_tensor, feature_map=True, cat_cls=False)
                ret = ret.permute(0, 2, 3, 1)
                ret = ret.reshape(B, -1, ret.size(-1)) # B, patch_num**2, dim
                
                return ret




