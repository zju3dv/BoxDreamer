from .base import PretrainedModelWrapper
from torchvision import models
import torch
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]
class DinoV2Wrapper(PretrainedModelWrapper):
    def __init__(self, ckpt_path: None, cfg):
        super().__init__(model_name_or_path='dinov2')
        self.ckpt_path = ckpt_path
        # todo: support local path loading
        self.model_type = cfg.get('model_type', 'dinov2_vits14_reg')
        assert self.model_type in ['dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']
        
        self.freeze = cfg.get('freeze', True)
        self.device = None
        
        self.load_model()
    
    def get_device(self):
        return self.device
    
    def to_device(self, device):
        self.model = self.model.to(device)
        self.device = device
    
    def load_model(self, device='cuda'):
        self.device = device
        # load backbone model
        
        if self.ckpt_path is not None:
            print(f'Loading model from {self.ckpt_path}')
            # load from local path
            model = torch.hub.load(self.ckpt_path, self.model_type, trust_repo=True, source='local').to(device)
        else:
            model = torch.hub.load('facebookresearch/dinov2', self.model_type).to(device)
            
        self.model = model
        
        if self.freeze:
            self.model.eval()
            # freeze the model
            for param in self.model.parameters():
                param.requires_grad = False
                
    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - torch.tensor(_RESNET_MEAN, device=img.device, requires_grad=False).view(1, 3, 1, 1)) / torch.tensor(_RESNET_STD, device=img.device, requires_grad=False).view(1, 3, 1, 1)
     
    def predict(self, input_tensor):
        flag = False
        if input_tensor.dim() == 5:
            # BTCHW
            B, T, C, H, W = input_tensor.size()
            input_tensor = input_tensor.flatten(0, 1)
            flag = True
        with torch.no_grad():
            input_tensor = self._resnet_normalize_image(input_tensor)
            ret = self.model.forward_features(input_tensor)['x_norm_patchtokens']
            if flag:
                ret = ret.view(B, T, *ret.shape[1:])
            
            return ret