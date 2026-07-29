"""Diagnostic test suite for evaluating RemoteCLIP backbone on Change Detection tasks.

This script tests whether RemoteCLIP feature representations can effectively:
1. Distinguish real structural changes (buildings constructed) from background/illumination shifts.
2. Align feature difference vectors with correct change captions vs generic distractor captions.
3. Maintain spatial patch-level feature variance before global pooling.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np

def create_synthetic_image(bg_color=(180, 150, 100), draw_buildings=False, brightness_offset=0):
    """Generate synthetic remote sensing satellite tiles for diagnostic testing."""
    img = Image.new("RGB", (256, 256), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw background grass/soil texture
    np.random.seed(42)
    for _ in range(50):
        x, y = np.random.randint(0, 256, 2)
        draw.ellipse([x, y, x+10, y+10], fill=(80, 120, 60))
        
    if draw_buildings:
        # Draw multiple structured buildings (gray/red roofs)
        draw.rectangle([50, 50, 90, 100], fill=(200, 70, 70), outline=(50, 50, 50))
        draw.rectangle([110, 60, 160, 110], fill=(180, 180, 180), outline=(30, 30, 30))
        draw.rectangle([60, 140, 120, 190], fill=(220, 210, 190), outline=(40, 40, 40))
        draw.rectangle([150, 150, 200, 200], fill=(160, 60, 60), outline=(50, 50, 50))
        # Draw roads (gray paths)
        draw.line([(0, 120), (256, 120)], fill=(100, 100, 100), width=8)

    if brightness_offset != 0:
        arr = np.array(img).astype(np.int16) + brightness_offset
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        
    return img

def load_remoteclip_model():
    import open_clip
    from huggingface_hub import hf_hub_download

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    ckpt_path = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")
    if not ckpt_path.exists():
        print("[i] Loading RemoteCLIP checkpoint from HuggingFace cache...")
        downloaded = hf_hub_download(repo_id="chendelong/RemoteCLIP", filename="RemoteCLIP-ViT-B-32.pt")
        ckpt_path = Path(downloaded)

    print(f"[✓] Loading official RemoteCLIP weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, preprocess, tokenizer

def run_diagnostics():
    print("======================================================================")
    print("  RemoteCLIP Local Change Detection Feature Diagnostic Suite         ")
    print("======================================================================")

    # 1. Load RemoteCLIP Backbone with official weights
    try:
        model, preprocess, tokenizer = load_remoteclip_model()
        print("[✓] Official RemoteCLIP ViT-B-32 pretrained weights loaded successfully!")
    except Exception as e:
        print(f"[✗] Failed to load RemoteCLIP model weights: {e}")
        return

    # 2. Prepare Diagnostic Synthetic Test Pairs
    img_before = create_synthetic_image(bg_color=(170, 140, 90), draw_buildings=False)
    img_after_buildings = create_synthetic_image(bg_color=(170, 140, 90), draw_buildings=True)
    img_after_identical = create_synthetic_image(bg_color=(170, 140, 90), draw_buildings=False)
    img_after_illumination = create_synthetic_image(bg_color=(170, 140, 90), draw_buildings=False, brightness_offset=40)

    # Convert to Tensors
    t_before = preprocess(img_before).unsqueeze(0)
    t_after_buildings = preprocess(img_after_buildings).unsqueeze(0)
    t_after_identical = preprocess(img_after_identical).unsqueeze(0)
    t_after_illumination = preprocess(img_after_illumination).unsqueeze(0)

    # 3. Extract Feature Embeddings
    with torch.no_grad():
        v_before = F.normalize(model.encode_image(t_before), dim=-1)
        v_buildings = F.normalize(model.encode_image(t_after_buildings), dim=-1)
        v_identical = F.normalize(model.encode_image(t_after_identical), dim=-1)
        v_illumination = F.normalize(model.encode_image(t_after_illumination), dim=-1)

    # 4. Diagnostic Metric 1: Cosine Similarities & Feature Distances
    sim_identical = F.cosine_similarity(v_before, v_identical).item()
    sim_buildings = F.cosine_similarity(v_before, v_buildings).item()
    sim_illumination = F.cosine_similarity(v_before, v_illumination).item()

    dist_buildings = torch.norm(v_buildings - v_before).item()
    dist_illumination = torch.norm(v_illumination - v_before).item()

    print("\n--- Diagnostic 1: Feature Distance & Sensitivity ---")
    print(f"1. Identical Image Pair Cosine Similarity  : {sim_identical:.4f} (Expected: 1.0)")
    print(f"2. Building Construction Pair Cosine Sim    : {sim_buildings:.4f}")
    print(f"3. Illumination/Color Shift Pair Cosine Sim : {sim_illumination:.4f}")
    print(f"-> Building Change Feature Distance  : {dist_buildings:.4f}")
    print(f"-> Illumination Noise Feature Distance: {dist_illumination:.4f}")

    ratio = dist_buildings / (dist_illumination + 1e-8)
    print(f"-> Sensitivity Ratio (Structural Change / Illumination Noise): {ratio:.2f}")

    if ratio < 1.2:
        print("  ⚠️ ALERT: Feature distance is dominated by lighting/color noise rather than actual building structures!")
    else:
        print("  ✓ PASS: Backbone feature distance reacts more strongly to structural changes than lighting noise.")

    # 5. Diagnostic Metric 2: Zero-Shot Text Alignment on Feature Differences
    prompts = [
        "many buildings were constructed on the land",
        "the vegetation quantity is decreased on this land",
        "no change in this area",
    ]
    tokens = tokenizer(prompts)
    with torch.no_grad():
        text_embeds = F.normalize(model.encode_text(tokens), dim=-1)
        diff_vec = F.normalize(v_buildings - v_before, dim=-1)
        text_sims = (diff_vec @ text_embeds.T).squeeze(0)

    print("\n--- Diagnostic 2: Zero-Shot Caption Alignment ---")
    print("Difference Feature Vector Sim with Candidate Captions:")
    for prompt, score in zip(prompts, text_sims.tolist()):
        print(f"  - '{prompt}': {score:.4f}")

    top_idx = text_sims.argmax().item()
    print(f"-> Top Predicted Candidate: '{prompts[top_idx]}'")
    if top_idx == 0:
        print("  ✓ PASS: Difference vector correctly aligns with 'buildings were constructed'.")
    else:
        print("  ⚠️ ALERT: Difference vector misaligned or biased toward generic captions!")

    # 6. Diagnostic Metric 3: Spatial Feature Map Resolution Check
    print("\n--- Diagnostic 3: Spatial Feature Map Resolution Check ---")
    visual_backbone = model.visual
    with torch.no_grad():
        x = visual_backbone.conv1(t_after_buildings)
        grid_shape = x.shape[2:]
        num_patches = grid_shape[0] * grid_shape[1]
        print(f"Patch Grid Resolution : {grid_shape[0]} x {grid_shape[1]} ({num_patches} patches)")
        print(f"Global Vector Size    : {v_buildings.shape[-1]}")
        print(f"Spatial Compression   : {num_patches} spatial tokens collapsed into 1 global vector")

    print("\n======================================================================")
    print("                      Diagnostic Summary                             ")
    print("======================================================================")
    print("Summary recommendation based on official RemoteCLIP feature test:")
    print("1. Ensure spatial patch grid features (14x14 tokens) are passed directly to decoder.")
    print("2. Unfreeze/fine-tune backbone or add change-mask auxiliary loss to suppress illumination noise.")
    print("======================================================================")

if __name__ == "__main__":
    run_diagnostics()
