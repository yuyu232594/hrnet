import torch


pretrained_dict=torch.load('/Users/wenyu/Desktop/TorchProject/HRnet/hrnet_w18_small_model_v1.pth',
                               map_location=torch.device('cpu'))
print(pretrained_dict.keys())