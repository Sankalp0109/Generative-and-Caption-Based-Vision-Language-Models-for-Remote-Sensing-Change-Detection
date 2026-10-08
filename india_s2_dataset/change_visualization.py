"""Explainable visual-change evidence for an aligned before/after pair."""
from collections import deque
import numpy as np

def joint_stretch(before, after, low=2, high=98):
    """Brighten both images with ONE shared scale: same range for all channels and both dates.

    Stretching each colour channel to its own range shifts the colour balance (green forest
    turned grey-violet, bare soil pink). One range for everything keeps true colour, and
    sharing it between dates keeps a real brightness difference (haze, a new roof) visible.
    All-zero (no-data) pixels are left out of the range.
    """
    combined = np.concatenate((before.reshape(-1, before.shape[-1]), after.reshape(-1, after.shape[-1])))
    combined = combined.astype("float32")
    valid = combined[np.any(combined > 0, axis=1)]
    lo, hi = np.nanpercentile(valid if len(valid) else combined, (low, high))
    scale = max(float(hi - lo), 1e-6)
    return (np.clip((before.astype("float32") - lo) / scale, 0, 1),
            np.clip((after.astype("float32") - lo) / scale, 0, 1))


def prepare_visual_evidence(before, after, percentile=75, stretch_low=2, stretch_high=98,
                            attention_radius=6, min_region_pixels=None, max_regions=8):
    """Build display-stretched views plus a change heatmap, attention map, and regions.

    percentile: attention percentile used as the region-detection cutoff (0-100);
        higher keeps only stronger, more confident evidence.
    stretch_low / stretch_high: percentile clip band used to contrast-stretch both
        images for display; does not affect the underlying aligned pixel values.
    attention_radius: half-window (pixels) of the box blur that turns the raw
        per-pixel heatmap into a smoother attention map; larger merges nearby
        change into fewer, more contiguous regions.
    min_region_pixels: minimum connected-component size kept as a region; None
        uses an adaptive default scaled to image size (max(40, pixel_count // 1000)).
    max_regions: cap on how many top-scoring regions are returned.
    """
    before_view, after_view = joint_stretch(before, after, stretch_low, stretch_high)
    heatmap = np.mean(np.abs(after_view-before_view), axis=2)
    radius = attention_radius; padded = np.pad(heatmap, radius, mode="reflect")
    integral = np.pad(padded, ((1,0),(1,0))).cumsum(0).cumsum(1); size = radius*2+1
    attention = (integral[size:,size:]-integral[:-size,size:]-integral[size:,:-size]+integral[:-size,:-size])/(size*size)
    attention = attention[:heatmap.shape[0],:heatmap.shape[1]]; attention /= max(float(np.nanmax(attention)), 1e-6)
    cutoff = float(np.nanpercentile(attention, percentile))
    minimum = min_region_pixels if min_region_pixels is not None else max(40, attention.size // 1000)
    regions = _regions(attention >= cutoff, attention, minimum)[:max_regions]
    return before_view, after_view, heatmap, attention, cutoff, regions

def _regions(mask, score, minimum):
    seen=np.zeros(mask.shape,bool); found=[]; height,width=mask.shape
    for row,column in zip(*np.nonzero(mask)):
        if seen[row,column]: continue
        queue=deque([(row,column)]); seen[row,column]=True; pixels=[]
        while queue:
            y,x=queue.popleft(); pixels.append((y,x))
            for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=ny<height and 0<=nx<width and mask[ny,nx] and not seen[ny,nx]: seen[ny,nx]=True; queue.append((ny,nx))
        if len(pixels)>=minimum:
            ys,xs=zip(*pixels); found.append({"x":min(xs),"y":min(ys),"width":max(xs)-min(xs)+1,"height":max(ys)-min(ys)+1,"pixels":len(pixels),"score":float(np.mean([score[y,x] for y,x in pixels]))})
    return sorted(found,key=lambda r:r["pixels"]*r["score"],reverse=True)
