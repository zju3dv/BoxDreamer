import torch
import imageio.v3 as iio 
from .base import TrackerWrapper
from src.utils.log import INFO
class CoTracker(TrackerWrapper):
    def __init__(self, ckpt_path: None, cfg=None):
        super().__init__(model_name_or_path='cotracker2')
        assert cfg is not None
        self.ckpt_path = ckpt_path
        
        self.grid_size = cfg['grid_size']
        self.freeze = cfg['freeze']
        self.device = None
        self.load_model()
        
    def get_device(self):
        return self.device
    
    def to_device(self, device):
        self.model = self.model.to(device)
    
    def load_model(self, device='cuda'):
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
        self.device = device
        INFO("Cotracker Model loaded")
        if self.freeze:
            self.model.eval()
            # freeze the model
            for param in self.model.parameters():
                param.requires_grad = False
            INFO("Cotracker Model freezed")
            
        
    def predict(self, input_tensor, queries=None, segm_mask=None):
        # input tensor shape should be B T C H W
        assert input_tensor.dim() == 5
        
        # offline mode supported
        pred_tracks, pred_visibility = self.model(input_tensor, grid_size=self.grid_size, segm_mask=segm_mask, queries=queries) # B T N 2,  B T N 1
        
        match = self._match(pred_tracks, pred_visibility)
        # make available track
        
        return match, torch.ones_like(match[..., 0]).bool()
    
    def _match(self, pred_tracks, pred_visibility):
        # this function is used to make B,T,N,2 matches which maintain the order of query_points and only keep the valid points
        # input shape B T N 2, B T N 1 
        # output shape B T N 2 (but only keep the valid points, N is not same as input)
        
        # select keypoints which is visible in all frames(in same batch)
        # use and operation to get the valid points
        matches = []
        
        for b in range(pred_tracks.shape[0]):
            batch_matches = []
            valid_points = pred_visibility[b][0]
            for t in range(pred_tracks.shape[1]):
                if (valid_points & pred_visibility[b][t]).sum() == 0:
                    continue
                valid_points = valid_points & pred_visibility[b][t]
                
            for t in range(pred_tracks.shape[1]):
                batch_matches.append(pred_tracks[b][t][valid_points])
            
            matches.append(batch_matches)
        
        # handle multi-batch case (padding)
        
        max_num = max([len(match[0]) for match in matches])
        
        # padding with known value
        for match in matches:
            # match : T N 2
            padding_len = max_num - len(match[0])
            if padding_len > 0:
                for i in range(len(match)):
                    # use last point to pad
                    match[i] = torch.cat([match[i], match[i][-1].repeat(padding_len, 1)], dim=0)

                
                
                
        matches = [torch.stack(match, dim=0) for match in matches]
            
        return torch.stack(matches, dim=0)
        
        
        
        

        