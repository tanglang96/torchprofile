import torch
from torch.nn.modules.transformer import Transformer
from torchprofile import profile_model

if __name__ == '__main__':
    embed_size = 512
    num_tokens = 30

    model = Transformer(embed_size)
    inputs = (
        torch.randn(num_tokens, 1, embed_size),
        torch.randn(num_tokens, 1, embed_size),
    )

    macs, peak_memory = profile_model(model, inputs)
    print('transformer, macs: %.2fM, peak_memory: %.2fKB'%(macs/1e6, peak_memory))
