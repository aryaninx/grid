#!/usr/bin/env python3
"""
app_fixed_sss_mag.py
--------------------
Fixed: Black boxes, progress bar, mag data support
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
from shapely.geometry import Point

st.set_page_config(page_title="Marine Survey Viewer", layout="wide", page_icon="🗺️")

# Initialize session state
if 'raster_layers' not in st.session_state:
    st.session_state.raster_layers = []
if 'vector_layers' not in st.session_state:
    st.session_state.vector_layers = []
if 'mag_points' not in st.session_state:
    st.session_state.mag_points = []

st.title("🗺️ Marine Survey Viewer")

# Sidebar
st.sidebar.header("📁 Load Data")

st.sidebar.header("⚡ Performance")
max_pixels = st.sidebar.select_slider(
    "Quality",
    options=[250, 500, 750, 1000, 1500, 2000],
    value=500
)

st.sidebar.header("🎨 Display")
basemap = st.sidebar.selectbox("Basemap", ['OpenStreetMap', 'Esri Satellite'], index=0)
mbes_colormap = st.sidebar.selectbox("MBES/Mag Colormap", ['ocean', 'viridis', 'terrain'], index=0)


def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive."""
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


@st.cache_data(show_spinner=False)
def tif_to_png_base64(file_path, colormap='gray', max_size=500, is_sss=False, is_mag=False):
    """
    Convert GeoTIFF to base64 PNG.
    
    FIX: Proper handling of SSS data to avoid black boxes
    """
    try:
        with rasterio.open(file_path) as src:
            orig_h, orig_w = src.height, src.width
            downsample = max(1, int(max(orig_h, orig_w) / max_size))
            out_h = orig_h // downsample
            out_w = orig_w // downsample
            
            # Read data
            if src.count >= 3 and is_sss:
                # RGB SSS
                data = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                for i in range(3):
                    data[:, :, i] = src.read(i + 1, out_shape=(out_h, out_w),
                                            resampling=rasterio.enums.Resampling.average)
                is_rgb = True
            elif src.count == 1 and is_sss:
                # Grayscale SSS - FIX: Proper normalization
                data = src.read(1, out_shape=(out_h, out_w),
                              resampling=rasterio.enums.Resampling.average)
                is_rgb = False
                # Convert to float BEFORE checking nodata
                data = data.astype(np.float32)
                
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
            else:
                # MBES or Mag
                data = src.read(1, out_shape=(out_h, out_w),
                              resampling=rasterio.enums.Resampling.average).astype(np.float32)
                is_rgb = False
                
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
            
            # Get WGS84 bounds
            bounds_wgs84 = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            
            # Create image
            if is_rgb:
                # RGB: Just normalize to 0-1
                img_data = np.clip(data.astype(np.float32) / 255.0, 0, 1)
                rgba = np.dstack([img_data, np.ones(img_data.shape[:2], dtype=np.float32)])
            else:
                # Grayscale: FIX - Proper contrast stretching
                valid_mask = np.isfinite(data)
                
                if not valid_mask.any():
                    return None, None
                
                valid_data = data[valid_mask]
                
                # FIX: Use appropriate percentiles for SSS vs MBES/Mag
                if is_sss:
                    # SSS: Usually 8-bit data, use tighter percentiles for contrast
                    vmin, vmax = np.percentile(valid_data, [1, 99])
                elif is_mag:
                    # Mag: Center around zero, use symmetric range
                    abs_max = np.percentile(np.abs(valid_data), 98)
                    vmin, vmax = -abs_max, abs_max
                else:
                    # MBES: Standard percentiles
                    vmin, vmax = np.percentile(valid_data, [2, 98])
                
                # Normalize
                if vmax > vmin:
                    data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
                else:
                    data_norm = np.zeros_like(data)
                
                # FIX: Set invalid data to 0, not NaN (prevents black boxes)
                data_norm[~valid_mask] = 0
                
                # Apply colormap
                cmap = get_cmap(colormap)
                rgba = cmap(data_norm)
                
                # FIX: Set alpha to 0 for invalid data (makes it transparent)
                rgba[:, :, 3] = valid_mask.astype(np.float32)
            
            # Convert to PNG
            rgba_uint8 = (rgba * 255).astype(np.uint8)
            img = Image.fromarray(rgba_uint8, mode='RGBA')
            
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            
            return img_base64, bounds_wgs84
    except Exception as e:
        st.error(f"Error processing raster: {e}")
        return None, None


def create_map(raster_layers, vector_layers, mag_points, basemap_choice):
    """Create Folium map."""
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
    
    # Add rasters
    for i, (img_base64, bounds) in enumerate(raster_layers):
        if img_base64 and bounds:
            img_url = f"data:image/png;base64,{img_base64}"
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.8,
                interactive=True,
                name=f"Raster {i+1}"
            ).add_to(m)
    
    # Add vectors
    for gdf, color, name in vector_layers:
        if gdf is not None:
            gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs and str(gdf.crs) != 'EPSG:4326' else gdf
            folium.GeoJson(gdf_wgs84, name=name,
                         style_function=lambda x, c=color: {'color': c, 'weight': 2}).add_to(m)
    
    # Add mag anomaly points
    if mag_points:
        for point in mag_points:
            # Color by magnitude
            magnitude = point.get('magnitude', 0)
            if magnitude > 50:
                color = 'red'
            elif magnitude > 20:
                color = 'orange'
            else:
                color = 'yellow'
            
            folium.CircleMarker(
                location=[point['lat'], point['lon']],
                radius=5,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                popup=f"<b>Mag Anomaly</b><br>Magnitude: {magnitude:.1f} nT",
                tooltip=f"{magnitude:.1f} nT"
            ).add_to(m)
    
    folium.plugins.Fullscreen().add_to(m)
    folium.plugins.MousePosition().add_to(m)
    folium.LayerControl().add_to(m)
    
    return m


# ==============================================================================
# DATA LOADING
# ==============================================================================

st.sidebar.subheader("📡 Sidescan (SSS)")

sss_ids = st.sidebar.text_area(
    "SSS File IDs (one per line)",
    height=150,
    placeholder="Paste all SSS file IDs here..."
)

if sss_ids and st.sidebar.button("🚀 Load All SSS", type="primary"):
    sss_id_list = [fid.strip() for fid in sss_ids.strip().split('\n') if fid.strip()]
    
    # FIX: Single progress container at top
    progress_container = st.container()
    
    with progress_container:
        st.info(f"📡 Loading {len(sss_id_list)} SSS tiles...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        
        for i, file_id in enumerate(sss_id_list):
            try:
                status_text.text(f"Downloading tile {i+1}/{len(sss_id_list)}...")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                    download_from_gdrive(file_id, tmp.name)
                    
                    # FIX: Explicitly mark as SSS
                    img, bounds = tif_to_png_base64(
                        tmp.name,
                        colormap='gray',
                        max_size=max_pixels,
                        is_sss=True  # This is the key!
                    )
                    
                    if img and bounds:
                        st.session_state.raster_layers.append((img, bounds))
                        success_count += 1
                    
                    os.unlink(tmp.name)
                
                progress_bar.progress((i + 1) / len(sss_id_list))
                
            except Exception as e:
                status_text.warning(f"Failed to load tile {i+1}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Loaded {success_count}/{len(sss_id_list)} SSS tiles!")

# MBES
st.sidebar.subheader("🗺️ Bathymetry (MBES)")

mbes_ids = st.sidebar.text_area(
    "MBES File IDs (one per line)",
    height=80
)

if mbes_ids and st.sidebar.button("Load MBES"):
    mbes_id_list = [fid.strip() for fid in mbes_ids.strip().split('\n') if fid.strip()]
    
    progress_container = st.container()
    
    with progress_container:
        st.info(f"🗺️ Loading {len(mbes_id_list)} MBES files...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file_id in enumerate(mbes_id_list):
            try:
                status_text.text(f"Downloading MBES {i+1}/{len(mbes_id_list)}...")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                    download_from_gdrive(file_id, tmp.name)
                    
                    img, bounds = tif_to_png_base64(
                        tmp.name,
                        colormap=mbes_colormap,
                        max_size=max_pixels,
                        is_sss=False,
                        is_mag=False
                    )
                    
                    if img and bounds:
                        st.session_state.raster_layers.append((img, bounds))
                    
                    os.unlink(tmp.name)
                
                progress_bar.progress((i + 1) / len(mbes_id_list))
                
            except Exception as e:
                status_text.warning(f"Failed: {e}")
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Loaded {len(mbes_id_list)} MBES files!")

# FIX: NEW - Magnetometer TIF
st.sidebar.subheader("🧲 Magnetometer (Gridded)")

mag_tif_ids = st.sidebar.text_area(
    "Mag TIF File IDs (one per line)",
    height=60,
    help="Gridded magnetometer data as GeoTIFF"
)

if mag_tif_ids and st.sidebar.button("Load Mag TIF"):
    mag_id_list = [fid.strip() for fid in mag_tif_ids.strip().split('\n') if fid.strip()]
    
    progress_container = st.container()
    
    with progress_container:
        st.info(f"🧲 Loading {len(mag_id_list)} mag files...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file_id in enumerate(mag_id_list):
            try:
                status_text.text(f"Downloading mag {i+1}/{len(mag_id_list)}...")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                    download_from_gdrive(file_id, tmp.name)
                    
                    img, bounds = tif_to_png_base64(
                        tmp.name,
                        colormap='seismic',  # Red-white-blue for mag data
                        max_size=max_pixels,
                        is_sss=False,
                        is_mag=True
                    )
                    
                    if img and bounds:
                        st.session_state.raster_layers.append((img, bounds))
                    
                    os.unlink(tmp.name)
                
                progress_bar.progress((i + 1) / len(mag_id_list))
                
            except Exception as e:
                status_text.warning(f"Failed: {e}")
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Loaded mag data!")

# FIX: NEW - Mag CSV targets
st.sidebar.subheader("🎯 Mag Targets (CSV)")

mag_csv_file = st.sidebar.file_uploader(
    "Upload Mag Targets CSV",
    type=['csv'],
    help="CSV with columns: lat, lon, magnitude (or similar)"
)

if mag_csv_file is not None:
    try:
        df = pd.read_csv(mag_csv_file)
        
        # Try to find lat/lon columns (flexible column names)
        lat_col = None
        lon_col = None
        mag_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'lat' in col_lower and lat_col is None:
                lat_col = col
            if 'lon' in col_lower and lon_col is None:
                lon_col = col
            if 'mag' in col_lower or 'anomaly' in col_lower or 'nt' in col_lower:
                mag_col = col
        
        if lat_col and lon_col:
            st.sidebar.success(f"✅ Found {len(df)} mag targets")
            
            # Optional: Filter by threshold
            if mag_col:
                mag_threshold = st.sidebar.slider(
                    "Show anomalies > (nT)",
                    min_value=0,
                    max_value=int(df[mag_col].max()) if len(df) > 0 else 100,
                    value=5
                )
                
                df_filtered = df[df[mag_col] > mag_threshold]
                st.sidebar.info(f"Showing {len(df_filtered)} anomalies > {mag_threshold}nT")
            else:
                df_filtered = df
                mag_col = None
            
            # Convert to points
            st.session_state.mag_points = []
            for _, row in df_filtered.iterrows():
                point = {
                    'lat': row[lat_col],
                    'lon': row[lon_col],
                    'magnitude': row[mag_col] if mag_col else 0
                }
                st.session_state.mag_points.append(point)
        else:
            st.sidebar.error("❌ Could not find lat/lon columns")
            st.sidebar.info(f"Available columns: {', '.join(df.columns)}")
    
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")

# Vectors
st.sidebar.subheader("📏 Vectors (GeoJSON)")

vector_id = st.sidebar.text_input("Vector File ID")
if vector_id and st.sidebar.button("Load Vector"):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
        try:
            download_from_gdrive(vector_id, tmp.name)
            gdf = gpd.read_file(tmp.name)
            st.session_state.vector_layers.append((gdf, 'yellow', 'Vector'))
            st.success(f"✅ Vector added ({len(gdf)} features)")
            os.unlink(tmp.name)
        except Exception as e:
            st.error(f"Error: {e}")

# Clear button
if st.sidebar.button("🗑️ Clear All"):
    st.session_state.raster_layers = []
    st.session_state.vector_layers = []
    st.session_state.mag_points = []
    st.success("Cleared!")
    st.rerun()

# Display map
if st.session_state.raster_layers or st.session_state.vector_layers or st.session_state.mag_points:
    st.subheader("🗺️ Survey Map")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Raster Layers", len(st.session_state.raster_layers))
    with col2:
        st.metric("Vector Layers", len(st.session_state.vector_layers))
    with col3:
        st.metric("Mag Targets", len(st.session_state.mag_points))
    with col4:
        est_mb = len(st.session_state.raster_layers) * 2
        st.metric("Est. Size", f"{est_mb} MB")
    
    m = create_map(st.session_state.raster_layers, st.session_state.vector_layers,
                  st.session_state.mag_points, basemap)
    st_folium(m, width=1400, height=800, key="main_map")
else:
    st.info("👆 Load data using sidebar")
    
    st.markdown("""
    ## 🗺️ Marine Survey Viewer
    
    ### Data Types Supported:
    
    **Rasters:**
    - 📡 SSS (Sidescan Sonar) - Load all 44 tiles at once
    - 🗺️ MBES (Bathymetry) - High-resolution seabed
    - 🧲 Mag TIF (Gridded magnetometer) - Anomaly maps
    
    **Points:**
    - 🎯 Mag Targets CSV - Individual anomalies with threshold filter
    
    **Vectors:**
    - 📏 SBP lines, cable routes, infrastructure
    
    ### Quick Start:
    1. Paste all SSS File IDs → Load All SSS
    2. Upload mag_targets.csv → Auto-display points
    3. Adjust threshold to show >5nT, >10nT, etc.
    """)
