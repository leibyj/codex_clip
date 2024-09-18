import torch

def z_score_normalize(img, clip_range=None):
    mean = img.mean(dim=(1, 2), keepdim=True)
    std = img.std(dim=(1, 2), keepdim=True)
    
    normalized_img = (img - mean) / (std + 1e-6) 
    
    if clip_range is not None:
        normalized_img = torch.clamp(normalized_img, clip_range[0], clip_range[1])

    return normalized_img

def min_max_normalize(img):
    min_val = torch.amin(img, dim=(1, 2), keepdim=True)
    max_val = torch.amax(img, dim=(1, 2), keepdim=True)
    
    normalized_img = (img - min_val) / (max_val - min_val + 1e-6)  

    return normalized_img

class MinMaxNormalize:
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-6)
        return normalized_tensor