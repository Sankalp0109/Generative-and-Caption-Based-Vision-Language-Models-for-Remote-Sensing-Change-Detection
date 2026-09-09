"""
Fetch a bi-temporal Sentinel-2 image pair over an Indian AOI using the
Microsoft Planetary Computer STAC API (no login/API key required).

This is a minimal proof-of-concept for the "automated dataset collection"
step of the change-captioning thesis pipeline: given an AOI and two date
ranges, pull the least-cloudy Sentinel-2 L2A scene in each window, clip to
the AOI, and save true-color PNGs for the "before" and "after" images.
"""
import io
import numpy as np
import planetary_computer
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from pystac_client import Client
from PIL import Image

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# AOI: GIFT City, Gandhinagar, Gujarat -- built up almost entirely from
# empty farmland since ~2012, so it's a strong visible-change example.
BBOX = [72.655, 23.145, 72.700, 23.185]  # lon_min, lat_min, lon_max, lat_max

WINDOWS = {
    "before_2016": ("2016-01-01", "2016-06-30"),
    "after_2024": ("2024-01-01", "2024-06-30"),
}


def candidate_items(catalog, date_range):
    search = catalog.search(
        collections=[COLLECTION],
        bbox=BBOX,
        datetime=f"{date_range[0]}/{date_range[1]}",
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = list(search.items())
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    return items


def try_clip(item, bbox, out_path, size=512):
    """Return True and save the PNG if the clipped window actually has data
    (some Sentinel-2 granules have diagonal no-data swath edges, so the AOI's
    bbox can fall outside the real footprint even though it's inside the
    scene's rectangular bbox)."""
    signed = planetary_computer.sign(item)
    href = signed.assets["visual"].href  # pre-rendered TCI (true color) asset
    with rasterio.open(href) as src:
        proj_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(*proj_bbox, transform=src.transform)
        data = src.read([1, 2, 3], window=window, out_shape=(3, size, size), resampling=rasterio.enums.Resampling.bilinear)
    if (data > 0).mean() < 0.99:  # mostly no-data pixels -> reject this scene
        return False
    arr = np.transpose(data, (1, 2, 0))
    Image.fromarray(arr).save(out_path)
    print(f"saved {out_path}  (scene date: {item.properties['datetime']}, cloud%: {item.properties.get('eo:cloud_cover')})")
    return True


def main():
    catalog = Client.open(STAC_URL)
    for label, date_range in WINDOWS.items():
        items = candidate_items(catalog, date_range)
        for item in items:
            if try_clip(item, BBOX, f"{label}.png"):
                break
        else:
            print(f"no usable (cloud-free, data-covered) scene found for {label} in {date_range}")


if __name__ == "__main__":
    main()
