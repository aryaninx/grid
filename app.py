#!/usr/bin/env python3
"""
app.py - Marine Survey Viewer (Cloud Version)
----------------------------------------------
Loads data from Google Drive or GitHub for cloud deployment.
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

st.set_page_config(page_title="Marine Survey Viewer", layout="wide", page_icon="🗺️")

st.title("🗺️ Marine Survey Viewer")

# Sidebar
st.sidebar.header("📁 Data Source")

data_source = st.sidebar.radio(
    "Load data from:",
    ["Sample Data (GitHub)", "Upload Files", "Google Drive Links"]
)

st.sidebar.header("⚡ Performance")
max_pixels = st.sidebar.select_slider(
    "Image Quality",
    options=[250, 500, 750, 1000],
    value=500
)

st.sidebar.header("🎨 Display")
basemap = st.sidebar.selectbox(
    "Basemap",
    ['OpenStreetMap', 'Esri Satellite', 'CartoDB Positron'],
    index=0
)

colormap = st.sidebar.selectbox(
    "Colormap",
    ['ocean', 'viridis', 'terrain', 'gray'],
    index=0
)


# ============================================================================
# CLOUD DATA LOADING FUNCTIONS
# ============================================================================

def download_from_gdrive(file_id, output_path):
    """
    Download file from Google Drive.
    
    File ID is from the shareable link:
    https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
    """
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    st.info(f"Downloading from Google Drive...")
    
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Handle large file confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(url, params=params, stream=True)
    
    # Download with progress
    total_size = int(response.headers.get('content-length', 0))
    
    if total_size > 0:
        progress_bar = st.progress(0)
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress_bar.progress(min(downloaded / total_size, 1.0))
        
        progress_bar.empty()
    else:
        with open(output_path, 'wb') as f:
            f.write(response.content)
    
    st.success(f"✅ Downloaded successfully")
    return output_path


def load_sample_data():
    """Load sample data from GitHub repo."""
    # This would load from data/sample/ folder in your repo
    sample_files = {
        'mbes': [],
        'sss': [],
        'vectors': []
    }
    
    # Check for sample files
    sample_dir = 'data/sample'
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if file.endswith('.tif'):
                if 'mbes' in file.lower():
                    sample_files['mbes'].append(os.path.join(sample_dir, file))
                elif 'sss' in file.lower():
                    sample_files['sss'].append(os.path.join(sample_dir, file))
            elif file.endswith('.geojson'):
                sample_files['vectors'].append(os.path.join(sample_dir, file))
    
    return sample_files


# ============================================================================
# RASTER PROCESSING FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def tif_to_png_base64(file_path_or_bytes, is_path=True, colormap='viridis', 
                      max_size=500, is_sss=False):
    """Convert GeoTIFF to base64 PNG."""
    
    # Handle both file path and bytes
    if is_path:
        tif_path = file_path_or_bytes
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            tmp.write(file_path_or_bytes)
            tif_path = tmp.name
    
    try:
        with rasterio.open(tif_path) as src:
            # Calculate downsample
            orig_h, orig_w = src.height, src.width
            downsample = max(1, int(max(orig_h, orig_w) / max_size))
            
            out_h = orig_h // downsample
            out_w = orig_w // downsample
            
            # Read data at lower resolution
            if src.count >= 3 and is_sss:
                data = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                for i in range(3):
                    data[:, :, i] = src.read(
                        i + 1,
                        out_shape=(out_h, out_w),
                        resampling=rasterio.enums.Resampling.average
                    )
                is_rgb = True
            else:
                data = src.read(
                    1,
                    out_shape=(out_h, out_w),
                    resampling=rasterio.enums.Resampling.average
                ).astype(np.float32)
                is_rgb = False
                
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
            
            # Get WGS84 bounds
            bounds_wgs84 = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            
            # Create image
            if is_rgb:
                img_data = np.clip(data.astype(np.float32) / 255.0, 0, 1)
                rgba = np.dstack([img_data, np.ones(img_data.shape[:2], dtype=np.float32)])
            else:
                valid_mask = np.isfinite(data)
                if not valid_mask.any():
                    return None, None
                
                valid_data = data[valid_mask]
                vmin, vmax = np.percentile(valid_data, [2, 98])
                
                data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
                data_norm[~valid_mask] = 0
                
                cmap = get_cmap(colormap)
                rgba = cmap(data_norm)
                rgba[:, :, 3] = valid_mask.astype(np.float32)
            
            # Convert to PNG
            rgba_uint8 = (rgba * 255).astype(np.uint8)
            img = Image.fromarray(rgba_uint8, mode='RGBA')
            
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            
            return img_base64, bounds_wgs84
    
    finally:
        if not is_path:
            try:
                os.unlink(tif_path)
            except:
                pass


def create_map(raster_layers, vector_layers, basemap_choice):
    """Create Folium map."""
    
    # Calculate center
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
    
    # Create map
    if basemap_choice == 'Esri Satellite':
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)
        folium.TileLayer(tiles=tiles, attr='Esri').add_to(m)
    elif basemap_choice == 'CartoDB Positron':
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')
    
    # Add raster layers
    for img_base64, bounds in raster_layers:
        if img_base64 and bounds:
            img_url = f"data:image/png;base64,{img_base64}"
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.7,
                interactive=True
            ).add_to(m)
    
    # Add vector layers
    for gdf, color, name in vector_layers:
        if gdf is not None:
            gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs else gdf
            folium.GeoJson(
                gdf_wgs84,
                name=name,
                style_function=lambda x, c=color: {'color': c, 'weight': 2}
            ).add_to(m)
    
    folium.plugins.Fullscreen().add_to(m)
    folium.plugins.MousePosition().add_to(m)
    folium.LayerControl().add_to(m)
    
    return m


# ============================================================================
# MAIN APP
# ============================================================================

raster_layers = []
vector_layers = []

# ============================================================================
# DATA LOADING BASED ON SOURCE
# ============================================================================

if data_source == "Sample Data (GitHub)":
    st.info("Loading sample data from repository...")
    
    sample_files = load_sample_data()
    
    if not any(sample_files.values()):
        st.warning("⚠️ No sample data found. Upload files or use Google Drive links.")
    else:
        # Process MBES
        for mbes_file in sample_files['mbes']:
            img, bounds = tif_to_png_base64(
                mbes_file, 
                is_path=True,
                colormap=colormap,
                max_size=max_pixels
            )
            if img:
                raster_layers.append((img, bounds))
        
        # Process SSS
        for sss_file in sample_files['sss']:
            img, bounds = tif_to_png_base64(
                sss_file,
                is_path=True,
                colormap='gray',
                max_size=max_pixels,
                is_sss=True
            )
            if img:
                raster_layers.append((img, bounds))
        
        # Process vectors
        for vector_file in sample_files['vectors']:
            gdf = gpd.read_file(vector_file)
            name = os.path.basename(vector_file)
            color = 'yellow' if 'sbp' in name.lower() else 'cyan'
            vector_layers.append((gdf, color, name))

elif data_source == "Upload Files":
    st.sidebar.subheader("Upload Files")
    
    uploaded_tifs = st.sidebar.file_uploader(
        "GeoTIFFs",
        type=['tif', 'tiff'],
        accept_multiple_files=True
    )
    
    uploaded_vectors = st.sidebar.file_uploader(
        "GeoJSON",
        type=['geojson', 'json'],
        accept_multiple_files=True
    )
    
    if uploaded_tifs:
        progress = st.progress(0)
        for i, uploaded in enumerate(uploaded_tifs):
            img, bounds = tif_to_png_base64(
                uploaded.getvalue(),
                is_path=False,
                colormap=colormap,
                max_size=max_pixels
            )
            if img:
                raster_layers.append((img, bounds))
            progress.progress((i + 1) / len(uploaded_tifs))
        progress.empty()
    
    if uploaded_vectors:
        for uploaded in uploaded_vectors:
            gdf = gpd.read_file(uploaded)
            color = 'yellow' if 'sbp' in uploaded.name.lower() else 'cyan'
            vector_layers.append((gdf, color, uploaded.name))

elif data_source == "Google Drive Links":
    st.sidebar.subheader("Google Drive File IDs")
    
    st.sidebar.info("""
    Get File ID from Google Drive link:
    `https://drive.google.com/file/d/FILE_ID/view`
    
    Make sure files are set to "Anyone with link can view"
    """)
    
    # MBES
    mbes_id = st.sidebar.text_input("MBES File ID")
    if mbes_id and st.sidebar.button("Load MBES"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            download_from_gdrive(mbes_id, tmp.name)
            img, bounds = tif_to_png_base64(
                tmp.name,
                is_path=True,
                colormap=colormap,
                max_size=max_pixels
            )
            if img:
                raster_layers.append((img, bounds))
            os.unlink(tmp.name)
    
    # SSS
    sss_ids = st.sidebar.text_area(
        "SSS File IDs (one per line)",
        height=100
    )
    if sss_ids and st.sidebar.button("Load SSS"):
        for file_id in sss_ids.strip().split('\n'):
            file_id = file_id.strip()
            if file_id:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                    download_from_gdrive(file_id, tmp.name)
                    img, bounds = tif_to_png_base64(
                        tmp.name,
                        is_path=True,
                        colormap='gray',
                        max_size=max_pixels,
                        is_sss=True
                    )
                    if img:
                        raster_layers.append((img, bounds))
                    os.unlink(tmp.name)
    
    # Vectors
    vector_id = st.sidebar.text_input("Vector File ID (GeoJSON)")
    if vector_id and st.sidebar.button("Load Vector"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
            download_from_gdrive(vector_id, tmp.name)
            gdf = gpd.read_file(tmp.name)
            vector_layers.append((gdf, 'yellow', 'Vector Layer'))
            os.unlink(tmp.name)

# ============================================================================
# DISPLAY MAP
# ============================================================================

if raster_layers or vector_layers:
    st.subheader("🗺️ Survey Map")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Raster Layers", len(raster_layers))
    with col2:
        st.metric("Vector Layers", len(vector_layers))
    
    m = create_map(raster_layers, vector_layers, basemap)
    st_folium(m, width=1400, height=800)

else:
    st.info("👆 Select a data source and load files")
    
    st.markdown("""
    ## 🗺️ Marine Survey Viewer
    
    ### Data Sources:
    
    **1. Sample Data (GitHub)**
    - Loads files from `data/sample/` folder in repository
    - Perfect for demos with small datasets
    
    **2. Upload Files**
    - Upload files directly from your computer
    - Works but limited by browser memory
    
    **3. Google Drive Links** ⭐
    - Store large files in Google Drive (15GB free!)
    - App downloads on-demand
    - Best for large datasets
    
    ### How to Use Google Drive:
    
    1. Upload your TIF/GeoJSON files to Google Drive
    2. Right-click → Share → "Anyone with link can view"
    3. Copy the File ID from the link
    4. Paste into the sidebar
    5. Click Load!
    
    Example link:
    ```
    https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view
                                  ^^^^^^^^^^^^^^^^
                                  This is the File ID
    ```
    """)
