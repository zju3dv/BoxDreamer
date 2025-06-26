import torch

class MatcherWrapper:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.model = None

    def load_model(self, device='cuda'):
        raise NotImplementedError("Subclasses should implement this method")

    def predict(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            return self.model(input_tensor)