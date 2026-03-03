#!/usr/bin/env python3
"""
app_debug_sss.py
----------------
DEBUG VERSION - Shows exactly what's wrong with SSS and Mag files
"""

import streamlit as st
import rasterio
from rasterio.warp import transform_bounds
import geopandas as gpd
import numpy as np
from PIL import Image
import folium
from streamlit_folium import st_folium
import tempfile
import os
import base64
from io import BytesIO
import requests
import matplotlib
matplotlib.use('Agg')
from matplotlib.cm import get_cmap
import pandas as pd

st.set_page_config(page_title="DEBUG MODE", layout="wide", page_icon="🐛")

if 'raster_layers' not in st.session_state:
    st.session_state.raster_layers = []
if 'vector_layers' not in st.session_state:
    st.session_state.vector_layers = []
if 'mag_points' not in st.session_state:
    st.session_state.mag_points = []

st.title("🐛 DEBUG MODE - SSS & Mag Diagnosis")

st.sidebar.header("📁 Debug Tools")

quality_preset = st.sidebar.radio(
    "Quality",
    ["Fast (500px)", "Good (1000px)", "High (2000px)", "Full Resolution"],
    index=1
)

quality_map = {
    "Fast (500px)": 500,
    "Good (1000px)": 1000,
    "High (2000px)": 2000,
    "Full Resolution": 10000
}
max_pixels = quality_map[quality_preset]

basemap = st.sidebar.selectbox("Basemap", ['OpenStreetMap', 'Esri Satellite'], index=0)


def download_from_gdrive(file_id, output_path):
    """Download from Google Drive."""
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    session = requests.Session()
    response = session.get(url, stream=True)
    
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(url, params=params, stream=True)
    
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    return output_path


def debug_raster(file_path):
    """
    Show EVERYTHING about the raster file.
    """
    st.subheader("🔍 Raster Debug Info")
    
    try:
        with rasterio.open(file_path) as src:
            # Basic info
            st.write("**Basic Info:**")
            st.json({
                "Format": src.driver,
                "Width": src.width,
                "Height": src.height,
                "Bands": src.count,
                "Data Type": str(src.dtypes[0]),
                "CRS": str(src.crs),
                "NoData": src.nodata
            })
            
            # Read band 1
            data = src.read(1)
            
            st.write("**Data Statistics (Band 1):**")
            st.json({
                "Min": float(data.min()),
                "Max": float(data.max()),
                "Mean": float(data.mean()),
                "Dtype": str(data.dtype),
                "Shape": data.shape,
                "Unique values (first 20)": sorted([int(x) for x in np.unique(data)[:20]])
            })
            
            # Check for nodata patterns
            zero_count = np.sum(data == 0)
            max_count = np.sum(data == 255)
            
            st.write("**Nodata Analysis:**")
            st.json({
                "Zeros (potential nodata)": f"{zero_count} ({zero_count/data.size*100:.1f}%)",
                "255s (potential nodata)": f"{max_count} ({max_count/data.size*100:.1f}%)",
                "Declared nodata value": src.nodata
            })
            
            # Show histogram
            st.write("**Value Distribution:**")
            hist, bins = np.histogram(data[data > 0], bins=50)
            st.bar_chart(pd.DataFrame({'count': hist}, index=bins[:-1].astype(int)))
            
            return True
            
    except Exception as e:
        st.error(f"Failed to open: {e}")
        
        # Try to read file type
        with open(file_path, 'rb') as f:
            header = f.read(100)
            st.write("**File Header (first 100 bytes):**")
            st.code(header.hex())
            
            # Check if it's actually a TIF
            if header[:2] == b'II' or header[:2] == b'MM':
                st.info("✅ This IS a TIFF file (valid header)")
            else:
                st.error("❌ This is NOT a TIFF file!")
                st.write(f"Header starts with: {header[:10]}")
        
        return False


def tif_to_png_base64_debug(file_path, colormap='gray', max_size=500, is_sss=False):
    """
    Convert with MAXIMUM debugging.
    """
    st.write("---")
    st.write(f"**Processing: {os.path.basename(file_path)}**")
    st.write(f"Mode: {'SSS' if is_sss else 'MBES'}")
    st.write(f"Max size: {max_size}px")
    
    try:
        with rasterio.open(file_path) as src:
            st.write(f"✅ File opened: {src.width}×{src.height}, {src.count} bands")
            
            orig_h, orig_w = src.height, src.width
            
            if max_size >= 10000:
                downsample = 1
                out_h, out_w = orig_h, orig_w
            else:
                downsample = max(1, int(max(orig_h, orig_w) / max_size))
                out_h = orig_h // downsample
                out_w = orig_w // downsample
            
            st.write(f"Downsample: {downsample}x → {out_w}×{out_h}")
            
            # Read data
            if src.count >= 3:
                st.write("📷 RGB mode detected")
                data = np.zeros((out_h, out_w, 3), dtype=np.float32)
                for i in range(3):
                    band = src.read(i + 1, out_shape=(out_h, out_w),
                                   resampling=rasterio.enums.Resampling.average)
                    data[:, :, i] = band.astype(np.float32)
                is_rgb = True
            else:
                st.write("🎚️ Grayscale mode detected")
                data = src.read(1, out_shape=(out_h, out_w),
                              resampling=rasterio.enums.Resampling.average)
                st.write(f"Raw data type: {data.dtype}")
                data = data.astype(np.float32)
                st.write(f"After conversion: {data.dtype}")
                is_rgb = False
            
            bounds_wgs84 = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            st.write(f"Bounds (WGS84): {bounds_wgs84}")
            
            nodata = src.nodata
            st.write(f"Declared nodata: {nodata}")
            
            if is_rgb:
                # RGB handling
                valid_mask = np.ones((out_h, out_w), dtype=bool)
                
                if nodata is not None:
                    invalid = np.all(data == nodata, axis=2)
                    valid_mask &= ~invalid
                    st.write(f"Pixels matching nodata={nodata}: {invalid.sum()}")
                
                black = np.all(data == 0, axis=2)
                valid_mask &= ~black
                st.write(f"Black pixels (0,0,0): {black.sum()}")
                
                white = np.all(data == 255, axis=2)
                valid_mask &= ~white
                st.write(f"White pixels (255,255,255): {white.sum()}")
                
                st.write(f"✅ Valid pixels: {valid_mask.sum()} / {valid_mask.size} ({valid_mask.sum()/valid_mask.size*100:.1f}%)")
                
                if not valid_mask.any():
                    st.error("❌ NO VALID DATA!")
                    return None, None
                
                data_norm = data / 255.0
                data_norm = np.clip(data_norm, 0, 1)
                
                rgba = np.zeros((out_h, out_w, 4), dtype=np.float32)
                rgba[:, :, :3] = data_norm
                rgba[:, :, 3] = valid_mask.astype(np.float32)
                
            else:
                # Grayscale
                valid_mask = np.ones(data.shape, dtype=bool)
                
                if nodata is not None:
                    invalid = (data == nodata)
                    valid_mask &= ~invalid
                    st.write(f"Pixels == nodata({nodata}): {invalid.sum()}")
                
                if is_sss:
                    zeros = (data == 0)
                    valid_mask &= ~zeros
                    st.write(f"Zero pixels (SSS nodata): {zeros.sum()}")
                    
                    maxvals = (data >= 255)
                    valid_mask &= ~maxvals
                    st.write(f"255 pixels (SSS nodata): {maxvals.sum()}")
                
                finite = np.isfinite(data)
                valid_mask &= finite
                st.write(f"Finite pixels: {finite.sum()}")
                
                st.write(f"✅ Valid pixels: {valid_mask.sum()} / {valid_mask.size} ({valid_mask.sum()/valid_mask.size*100:.1f}%)")
                
                if not valid_mask.any():
                    st.error("❌ NO VALID DATA!")
                    return None, None
                
                valid_data = data[valid_mask]
                st.write(f"Valid data range: {valid_data.min():.2f} to {valid_data.max():.2f}")
                
                # Calculate percentiles
                if is_sss:
                    vmin, vmax = np.percentile(valid_data, [0.5, 99.5])
                    st.write(f"SSS percentiles [0.5%, 99.5%]: {vmin:.2f}, {vmax:.2f}")
                else:
                    vmin, vmax = np.percentile(valid_data, [2, 98])
                    st.write(f"MBES percentiles [2%, 98%]: {vmin:.2f}, {vmax:.2f}")
                
                if vmax == vmin:
                    st.warning("⚠️ vmin == vmax, adding 1 to vmax")
                    vmax = vmin + 1
                
                data_norm = (data - vmin) / (vmax - vmin)
                data_norm = np.clip(data_norm, 0, 1)
                data_norm[~valid_mask] = 0
                
                st.write(f"Normalized range: {data_norm[valid_mask].min():.3f} to {data_norm[valid_mask].max():.3f}")
                
                cmap = get_cmap(colormap)
                rgba = cmap(data_norm)
                rgba[:, :, 3] = valid_mask.astype(np.float32)
                
                st.write(f"Alpha channel: min={rgba[:,:,3].min()}, max={rgba[:,:,3].max()}")
            
            # Create image
            rgba_uint8 = (rgba * 255).astype(np.uint8)
            st.write(f"Final image: {rgba_uint8.shape}, dtype={rgba_uint8.dtype}")
            
            img = Image.fromarray(rgba_uint8, mode='RGBA')
            
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            
            size_mb = len(img_base64) / 1024 / 1024
            st.success(f"✅ Created PNG: {size_mb:.2f} MB")
            
            return img_base64, bounds_wgs84
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None


def create_map(raster_layers, basemap_choice):
    """Create map."""
    all_bounds = [b for _, b in raster_layers if b]
    
    if all_bounds:
        min_lon = min(b[0] for b in all_bounds)
        min_lat = min(b[1] for b in all_bounds)
        max_lon = max(b[2] for b in all_bounds)
        max_lat = max(b[3] for b in all_bounds)
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
    else:
        center_lat, center_lon = 53.7, 2.5
    
    if basemap_choice == 'Esri Satellite':
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
        folium.TileLayer(tiles=tiles, attr='Esri').add_to(m)
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')
    
    for i, (img_base64, bounds) in enumerate(raster_layers):
        if img_base64 and bounds:
            img_url = f"data:image/png;base64,{img_base64}"
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.85,
                interactive=True,
                name=f"Raster {i+1}"
            ).add_to(m)
    
    folium.plugins.Fullscreen().add_to(m)
    folium.plugins.MousePosition().add_to(m)
    
    return m


# ==============================================================================
# DEBUG MODE
# ==============================================================================

st.sidebar.subheader("🐛 Debug Single SSS File")

debug_sss_id = st.sidebar.text_input("SSS File ID (just one)")

if debug_sss_id and st.sidebar.button("🔍 Debug SSS"):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        try:
            st.info("Downloading...")
            download_from_gdrive(debug_sss_id, tmp.name)
            
            file_size = os.path.getsize(tmp.name) / 1024 / 1024
            st.write(f"Downloaded: {file_size:.2f} MB")
            
            # First, debug the file
            if debug_raster(tmp.name):
                st.write("---")
                st.write("**Now attempting to process:**")
                
                # Then try to process it
                img, bounds = tif_to_png_base64_debug(
                    tmp.name,
                    colormap='gray',
                    max_size=max_pixels,
                    is_sss=True
                )
                
                if img and bounds:
                    st.success("✅ Processing succeeded!")
                    st.session_state.raster_layers = [(img, bounds)]
                else:
                    st.error("❌ Processing failed")
            
            os.unlink(tmp.name)
            
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

st.sidebar.subheader("🐛 Debug Mag File")

mag_file = st.sidebar.file_uploader("Upload Mag File", type=['tif', 'tiff', 'img', 'xyz', 'asc'])

if mag_file is not None and st.sidebar.button("🔍 Debug Mag"):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        try:
            tmp.write(mag_file.getvalue())
            tmp.flush()
            
            file_size = os.path.getsize(tmp.name) / 1024 / 1024
            st.write(f"File size: {file_size:.2f} MB")
            
            # Check if it's actually a TIF
            with open(tmp.name, 'rb') as f:
                header = f.read(10)
                st.write("**File header:**")
                st.code(header.hex())
                
                if header[:2] == b'II' or header[:2] == b'MM':
                    st.success("✅ Valid TIFF header")
                    debug_raster(tmp.name)
                else:
                    st.error("❌ NOT a TIFF file!")
                    st.write(f"First bytes: {header}")
                    st.write("This file is probably XYZ, ASCII, or another format.")
                    st.info("You need to convert XYZ to GeoTIFF first using GDAL")
            
            os.unlink(tmp.name)
            
        except Exception as e:
            st.error(f"Error: {e}")

# Display
if st.session_state.raster_layers:
    st.subheader("🗺️ Map (if processing succeeded)")
    m = create_map(st.session_state.raster_layers, basemap)
    st_folium(m, width=1400, height=600)

st.markdown("""
---
## 🐛 Debug Instructions

### For SSS Black Boxes:
1. Paste ONE SSS File ID
2. Click "Debug SSS"
3. Look at the output:
   - What's the data type?
   - What's the value range?
   - How many zeros/255s?
   - What % is valid data?

### For Mag TIF:
1. Upload your mag file
2. Click "Debug Mag"
3. Check if it's actually a TIF:
   - Valid TIFF header? 
   - Or is it XYZ/ASCII?

### Send me the output!
Screenshot the debug info and I'll tell you exactly what's wrong.
""")
