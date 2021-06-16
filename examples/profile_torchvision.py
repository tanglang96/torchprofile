import torch
from torchprofile import profile_model
from torchvision import models

if __name__ == '__main__':
    for name, model in models.__dict__.items():
        if not name.islower() or name.startswith('__') or not callable(model):
            continue

        model = model().eval()
        if 'inception' not in name:
            inputs = torch.randn(1, 3, 224, 224)
        else:
            inputs = torch.randn(1, 3, 299, 299)

        macs, peak_memory = profile_model(model, inputs)
        print('%s, macs: %.2fM, peak_memory: %.2fKB'%(name, macs/1e6, peak_memory))
