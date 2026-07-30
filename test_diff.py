import torch
from src.models.phase6.tile_difference import TileDifference

model = TileDifference(backbone_dim=512, fusion_dim=512, num_heads=4)
B, N, D = 2, 4, 512
before_features = torch.randn(B, N, D)
after_features = torch.randn(B, N, D)

diff = model(before_features, after_features)
print(diff.shape)
