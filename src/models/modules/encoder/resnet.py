from .base import PretrainedModelWrapper
from torchvision import models
import torch.nn as nn
import torch
class ResNetWrapper(PretrainedModelWrapper):
    def __init__(self, ckpt_path: None, cfg):
        super().__init__(model_name_or_path='resnet')
        
        self.ckpt_path = ckpt_path
        # todo: support local path loading
        
        self.model_type = cfg.model_type
        assert self.model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        
        self.activation = {}
        
        self.freeze = cfg.freeze
        self.device = None
        
        self.load_model()
        
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
    def get_device(self):
        return self.device
    
    def to_device(self, device):
        self.model = self.model.to(device)
        
        
    def load_model(self, device='cuda'):
        self.device = device
        # load resnet model based on model_type
        if self.model_type == 'resnet18':
            self.model = models.resnet18(pretrained=True).to(device)
        elif self.model_type == 'resnet34':
            self.model = models.resnet34(pretrained=True).to(device)
        elif self.model_type == 'resnet50':
            self.model = models.resnet50(pretrained=True).to(device)
        elif self.model_type == 'resnet101':
            self.model = models.resnet101(pretrained=True).to(device)
        elif self.model_type == 'resnet152':
            self.model = models.resnet152(pretrained=True).to(device)
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        # add hooks to get intermediate features
        self.model.layer4.register_forward_hook(self.get_activation('layer4'))
        
        self.model = self.model
        
        if self.freeze:
            self.model.eval()
            # freeze the model
            for param in self.model.parameters():
                param.requires_grad = False
        
    def predict(self, input_tensor):
        flag = False
        if input_tensor.dim() == 5:
            # BTCHW
            B, T, C, H, W = input_tensor.size()
            input_tensor = input_tensor.flatten(0, 1)
            flag = True
        
        with torch.no_grad():
            self.model(input_tensor)
            if flag:
                # B, T, C, H, W
                self.activation['layer4'] = self.activation['layer4'].view(B, T, *self.activation['layer4'].size()[1:]).flatten(3, 4).transpose(2, 3)
                
                return self.activation['layer4']
            else:
                return self.activation['layer4'].flatten(2, 3).transpose(1, 2)
        
        
        
        
    