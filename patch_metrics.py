import sys

with open("src/metrics.py", "r") as f:
    code = f.read()

# Add a print statement in generate_caption_greedy
new_code = code.replace(
    "temporal_dim = 1 if before_image.ndim == 4 else 0",
    "print(f'before_image.shape: {before_image.shape}'); temporal_dim = 1 if before_image.ndim == 4 else 0"
)

with open("src/metrics.py", "w") as f:
    f.write(new_code)
