import torch
from src.dataset import extract_ordered_patches_from_pil

# Wait, let's just trace how (1, 2, 4, 3, 224, 224) can happen.
# We know before_image is a tensor.
# temporal_dim = 1 if before_image.ndim == 4 else 0
# images = torch.stack([before_image, after_image], dim=temporal_dim).unsqueeze(0)

# Case 1: before_image is (4, 3, 224, 224)
b1 = torch.zeros(4, 3, 224, 224)
a1 = torch.zeros(4, 3, 224, 224)
temporal_dim = 1 if b1.ndim == 4 else 0
images = torch.stack([b1, a1], dim=temporal_dim).unsqueeze(0)
print(f"Case 1 (pre-patched): {images.shape}")
# images is (1, 4, 2, 3, 224, 224).
# tile_extractor returns images -> (1, 4, 2, 3, 224, 224). size(2) is 2. NO ERROR.

# Case 2: before_image is (3, 448, 448) (Not pre-patched)
b2 = torch.zeros(3, 448, 448)
a2 = torch.zeros(3, 448, 448)
temporal_dim = 1 if b2.ndim == 4 else 0
images = torch.stack([b2, a2], dim=temporal_dim).unsqueeze(0)
print(f"Case 2 (not patched): {images.shape}")
# images is (1, 2, 3, 448, 448).
# tile_extractor gets (1, 2, 3, 448, 448).
tiles = []
for r in range(0, 448, 224):
    for c in range(0, 448, 224):
        patch = images[:, :, :, r:r+224, c:c+224] # (1, 2, 3, 224, 224)
        tiles.append(patch)
tiles_tensor = torch.stack(tiles, dim=1)
print(f"Case 2 after TileExtractor: {tiles_tensor.shape}")
# tiles_tensor is (1, 4, 2, 3, 224, 224). size(2) is 2. NO ERROR.

# How on earth do we get (1, 2, 4, 3, 224, 224)?
# What if before_image is (4, 3, 224, 224) but temporal_dim was 0?
images = torch.stack([b1, a1], dim=0).unsqueeze(0)
print(f"Case 3 (pre-patched, temporal_dim=0): {images.shape}")
# images is (1, 2, 4, 3, 224, 224) !!!
# Then tile_extractor gets (1, 2, 4, 3, 224, 224).
# images.ndim == 6. Returns images!
# TileEncoder gets (1, 2, 4, 3, 224, 224). size(2) is 4 != 2. ERROR!

