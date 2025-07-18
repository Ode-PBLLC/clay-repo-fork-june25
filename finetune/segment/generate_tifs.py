# This script is used to generate image and label tifs for training and testing
import pyproj
from shapely.geometry import box, mapping
from shapely.ops import transform as shp_transform
import time
import pystac_client
import stackstac
from box import Box
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import os
import rioxarray
import pandas as pd
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from shapely.ops import unary_union
from shapely.geometry import Point
import rasterio
from rasterio.windows import from_bounds
import itertools
import dask
import gc

# Imagery start and end dates
start_date = "2022-01-01"
end_date = "2022-04-30"

# Directories
train_labels_dir = "data/mydata/train/labels"
train_images_dir = "data/mydata/train/chips"
test_labels_dir = "data/mydata/val/labels"
test_images_dir = "data/mydata/val/chips"

# STAC parameters
stac_api = "https://earth-search.aws.element84.com/v1"
collection = "sentinel-2-l2a"
platform = "sentinel-2-l2a"
bands_to_embed = ["blue","green","red","rededge1","rededge2","rededge3","nir", "nir08", "swir16","swir22"]
gsd = 10  
composite_method = "median"  # or "mean"
time_to_wait = 0.2  # seconds between requests

india_adm = gpd.read_file("data/mydata/gadm41_IND_1.json").set_crs(epsg=4326)
andhra_pradesh_bounds = india_adm[india_adm["NAME_1"]=="AndhraPradesh"]
dataset_name = "andhrapradesh"

# --- Chunk Clark Labs tif ---
clark_tif_path = "data/mydata/india_landcover_2022_v1exp.tif"  

# --- Piecemeal chip function using tile bounds ---
def save_raster_chips_from_tiles(raster_path, poly_gdf, out_dir, tile_size, prefix="chunk"):
    os.makedirs(out_dir, exist_ok=True)
    tiles = polygon_to_tiles(poly_gdf, tile_size)
    with rasterio.open(raster_path) as src:
        for i, bounds in enumerate(tiles):
            window = from_bounds(*bounds, transform=src.transform)
            chip = src.read(window=window)
            # Check if chip is the correct size
            if chip.shape[1] == tile_size and chip.shape[2] == tile_size:
                chip_meta = src.meta.copy()
                chip_meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": rasterio.windows.transform(window, src.transform)
                })
                chip_path = os.path.join(out_dir, f"{prefix}_chip_{i}.tif")
                #print(f"Saving chip: {chip_path}")
                with rasterio.open(chip_path, "w", **chip_meta) as dest:
                    dest.write(chip)

def save_xarray_chips_from_tiles(raster_path, poly_gdf, out_dir, tile_size, prefix="chunk", max_workers=8):
    os.makedirs(out_dir, exist_ok=True)
    # Open raster with rioxarray
    da = rioxarray.open_rasterio(raster_path)
    da_8857 = da.rio.reproject(dst_crs="EPSG:8857")

    # Use polygon_to_tiles to get valid chip bounds
    tiles = polygon_to_tiles(poly_gdf, tile_size)

    def process_chip(tile_bounds, chip_num):
        minx, miny, maxx, maxy = tile_bounds
        chip = da_8857.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        #print(chip.shape)
        if chip.rio.crs is None:
           chip = chip.riowrite_crs("EPSG:8857")  
        threshold_coverage = tile_size*tile_size*0.001 # Edit - Only grab chips with more than 0.1%
        #if chip.shape[1] == tile_size and chip.shape[2] == tile_size and chip.any():
        if chip[0].sum().values > threshold_coverage:
            chip_path = os.path.join(out_dir, f"{prefix}_chip_{chip_num}.tif")
            print(f"Saving chip: {chip_path}", flush=True)
            chip.rio.to_raster(chip_path)
        return chip_num

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_chip, tile_bounds, chip_num)
            for chip_num, tile_bounds in enumerate(tiles)
        ]
        with tqdm(total=len(futures), desc=f"Chipping {os.path.basename(raster_path)}") as pbar:
            for _ in as_completed(futures):
                pbar.update(1)
    del da, da_8857
    gc.collect()


def polygon_to_tiles(poly_gdf, tile_size):
    merged_geom = unary_union(poly_gdf.geometry)
    minx, miny, maxx, maxy = merged_geom.bounds
    tile_size_m = tile_size * gsd
    x_centers = np.arange(minx + tile_size_m / 2, maxx + tile_size_m / 2, tile_size_m)
    y_centers = np.arange(miny + tile_size_m / 2, maxy + tile_size_m / 2, tile_size_m)

    tiles = []
    for xc in x_centers:
        for yc in y_centers:
            if merged_geom.contains(Point(xc, yc)):
                tile_bounds = (
                    xc - tile_size_m // 2,
                    yc - tile_size_m // 2,
                    xc + tile_size_m // 2,
                    yc + tile_size_m // 2,
                )
                tiles.append(tile_bounds)
    return tiles

def process_geojson_and_chip(raster_path, geojson_path, out_dir, chip_size=2048, prefix="train"):
    print(f"Processing geojson: {geojson_path} for raster: {raster_path}")
    os.makedirs(out_dir, exist_ok=True)
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:8857")
    # Instead of looping over polygons, chip the raster using all polygons at once
    save_xarray_chips_from_tiles(
        raster_path=raster_path,
        poly_gdf=gdf,
        out_dir=out_dir,
        tile_size=chip_size,
        prefix=prefix
    )
    print(f"Finished processing geojson: {geojson_path}\n")

# Clip Clark Labs tif to Andhra Pradesh boundary before chipping

clipped_tif_path = "data/mydata/aoi_clipped.tif"
if not os.path.exists(clipped_tif_path):
    print("Clipping Clark Labs tif to Andhra Pradesh boundary and binarizing...")
    with rasterio.open(clark_tif_path) as src:
        ap_crs = src.crs 
        andhra_pradesh_bounds = andhra_pradesh_bounds.to_crs(ap_crs)
        out_image, out_transform = mask(src, andhra_pradesh_bounds.geometry, crop=True)
        # Binarize: set all values == 3 to 1, all others to 0
        # out_image shape: (bands, height, width)
        binarized = (out_image == 3).astype('uint8')
        print(f"  Binarized: unique values in output: {np.unique(binarized)}")
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": binarized.shape[1],
            "width": binarized.shape[2],
            "transform": out_transform,
            "dtype": 'uint8',
            "compress": "lzw"
        })
        with rasterio.open(clipped_tif_path, "w", **out_meta) as dest:
            dest.write(binarized)
    print(f"Saved clipped and binarized raster to {clipped_tif_path}")
else:
    print(f"Clipped raster already exists at {clipped_tif_path}")

# Use the clipped raster for all subsequent processing
clark_tif_path = clipped_tif_path

train_prefix = f"train_{dataset_name}"
test_prefix = f"val_{dataset_name}"

#Chunk label tif
# process_geojson_and_chip(
#     raster_path="data/mydata/india_landcover_2022_v1exp.tif",
#     geojson_path="data/mydata/train_geom.geojson",
#     out_dir="data/mydata/train/labels",
#     chip_size=256,
#     prefix=train_prefix
# )
# process_geojson_and_chip(
#     raster_path="data/mydata/india_landcover_2022_v1exp.tif",
#     geojson_path="data/mydata/test_geom.geojson",
#     out_dir="data/mydata/val/labels",
#     chip_size=256,
#     prefix=test_prefix
# )

# --- Generate and save image tif files based on those chunk polygons, using Sentinel 2 from STAC ---

def reproject_bounds_8857_to_4326(bounds_8857):
    minx, miny, maxx, maxy = bounds_8857
    proj_8857 = pyproj.CRS("EPSG:8857")
    proj_4326 = pyproj.CRS("EPSG:4326")
    transformer = pyproj.Transformer.from_crs(proj_8857, proj_4326, always_xy=True)
    poly_8857 = box(minx, miny, maxx, maxy)
    poly_4326 = shp_transform(transformer.transform, poly_8857)
    return poly_4326.bounds  # (lon_min, lat_min, lon_max, lat_max)

def composite_arrays(xarray_stack, method="median"):
    if method == "median":
        return xarray_stack.median(dim="time", skipna=True)
    elif method == "mean":
        return xarray_stack.mean(dim="time", skipna=True)
    else:
        raise ValueError("Unknown composite method")

def process_tile(bounds_8857, start_date, end_date, out_path, ref_chip_path=None):
    #print(f"Processing Sentinel-2 tile for bounds: {bounds_8857}")
    with dask.config.set(scheduler='threads', num_workers=1):
        latlon_bounds = reproject_bounds_8857_to_4326(bounds_8857)

        time.sleep(time_to_wait)
        catalog = pystac_client.Client.open(stac_api)
        search = catalog.search(
            collections=[collection],
            datetime=f"{start_date}/{end_date}",
            bbox=latlon_bounds,
            max_items=100,
            query={"eo:cloud_cover": {"lt": 80}},
        )

        items = search.item_collection()

        if not items:
            print(f"No items found for bounds: {latlon_bounds}")
            return
        if ref_chip_path is not None:
            ref_chip = rioxarray.open_rasterio(ref_chip_path)
            ref_crs = ref_chip.rio.crs
            ref_transform = ref_chip.rio.transform()
            ref_width = ref_chip.rio.width
            ref_height = ref_chip.rio.height

        xarray_stack = stackstac.stack(
            items,
            assets=bands_to_embed,
            bounds=bounds_8857,
            epsg=8857,
            snap_bounds=False,
            resolution=gsd,
            resampling=Resampling.nearest,
            dtype="float32",
            fill_value=np.float32(0),
            rescale=False,
        ).compute()

        xarray_stack = xarray_stack.where(xarray_stack != 0, np.nan)
        if "time" not in xarray_stack.dims or len(xarray_stack.time) == 0:
            print(f"No valid time slices for bounds: {bounds_8857}")
            return

        xarray_stack = xarray_stack.chunk({'time': 5, 'band': 5, 'x': 256, 'y': 256})
        composite = composite_arrays(xarray_stack, method=composite_method)
        composite = composite.compute()
        if "time" in composite.dims:
            composite = composite.isel(time=0, drop=True)
        composite = composite.assign_attrs(xarray_stack.attrs)
        print("composite complete")

        if ref_chip is not None:
            print(f"Reprojecting Sentinel-2 tile to match reference chip: {ref_chip_path}")

            # Ensure composite has CRS set
            if composite.rio.crs is None:
                composite = composite.riowrite_crs("EPSG:8857")  # or the correct source CRS

            # Reproject to match reference chip
            ref_chip = ref_chip.rio.reproject("EPSG:8857")
            composite = composite.rio.reproject_match(ref_chip,resampling=Resampling.nearest)
            
        print(f"    Saving Sentinel-2 chip: {out_path}")
        composite.rio.to_raster(out_path)
        del composite, xarray_stack, ref_chip
        gc.collect()

def process_all_chips(chip_dir, out_dir, start_date, end_date, max_workers=20):
    print(f"Processing all chips in {chip_dir} to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    
    #start_record = 2282
    #start_record = 129
    chip_files = [f for f in os.listdir(chip_dir) if f.endswith(".tif")] #[start_record:]
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chip_num, chip_file in enumerate(chip_files):
            chip_path = os.path.join(chip_dir, chip_file)
            try:
                with rioxarray.open_rasterio(chip_path) as src:
                    src = src.rio.reproject("EPSG:8857")
                    bounds = src.rio.bounds()
                out_path = os.path.join(out_dir, chip_file)
                futures.append(executor.submit(process_tile, bounds, start_date, end_date, out_path, chip_path))
            except Exception as e:
                print(f"Error opening {chip_file}: {e}")

        with tqdm(total=len(futures), desc="Processing chips") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()  # raise exceptions here
                except Exception as e:
                    print(f"Task failed: {e}")
                pbar.update(1)

    print(f"Finished processing all chips in {chip_dir}")


#process_all_chips(train_labels_dir, train_images_dir, start_date, end_date)
process_all_chips(test_labels_dir, test_images_dir, start_date, end_date)





