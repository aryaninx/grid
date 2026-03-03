#!/usr/bin/env python3
"""
app_fully_fixed.py
------------------
FIXES:
1. SSS black boxes - proper nodata handling + contrast stretching
2. Quality slider up to FULL resolution (5000px)
3. Mag TIF upload support (all raster formats)
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

# FIX 2: Extended quality slider with FULL resolution option
quality_preset = st.sidebar.radio(
    "Quality Preset",
    ["Fast (500px)", "Good (1000px)", "High (2000px)", "Ultra (3000px)", "Full Resolution"],
    index=1,
    help="Higher quality = slower loading. Use Full Resolution only for single tiles."
)

# Map preset to max_pixels
quality_map = {
    "Fast (500px)": 500,
    "Good (1000px)": 1000,
    "High (2000px)": 2000,
    "Ultra (3000px)": 3000,
    "Full Resolution": 10000  # Very high limit for full res
}
max_pixels = quality_map[quality_preset]

if quality_preset == "Full Resolution":
    st.sidebar.warning("⚠️ Full res: Use for 1-5 tiles only. Will be slow for 44 tiles!")

st.sidebar.caption(f"Max resolution: {max_pixels}×{max_pixels}px")

st.sidebar.header("🎨 Display")
basemap = st.sidebar.selectbox("Basemap", ['OpenStreetMap', 'Esri Satellite'], index=0)
mbes_colormap = st.sidebar.selectbox("MBES/Mag Colormap", ['ocean', 'viridis', 'terrain', 'seismic'], index=0)


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
    
    COMPLETE FIX for SSS black boxes:
    - Proper nodata detection
    - Aggressive contrast stretching
    - Correct alpha channel handling
    """
    try:
        with rasterio.open(file_path) as src:
            orig_h, orig_w = src.height, src.width
            
            # Calculate downsample factor
            if max_size >= 10000:  # Full resolution mode
                downsample = 1
                out_h, out_w = orig_h, orig_w
            else:
                downsample = max(1, int(max(orig_h, orig_w) / max_size))
                out_h = orig_h // downsample
                out_w = orig_w // downsample
            
            # FIX 1: CRITICAL - Read data with proper dtype
            if src.count >= 3:
                # RGB (common for SSS exports)
                data = np.zeros((out_h, out_w, 3), dtype=np.float32)
                for i in range(3):
                    band = src.read(i + 1, out_shape=(out_h, out_w),
                                   resampling=rasterio.enums.Resampling.average)
                    data[:, :, i] = band.astype(np.float32)
                is_rgb = True
            else:
                # Single band
                data = src.read(1, out_shape=(out_h, out_w),
                              resampling=rasterio.enums.Resampling.average)
                data = data.astype(np.float32)  # CRITICAL: Convert to float
                is_rgb = False
            
            # Get WGS84 bounds
            bounds_wgs84 = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            
            # FIX 1: CRITICAL - Identify nodata values properly
            nodata = src.nodata
            
            if is_rgb:
                # RGB handling
                # Check for nodata (often 0,0,0 or 255,255,255 in SSS)
                valid_mask = np.ones((out_h, out_w), dtype=bool)
                
                # Common SSS nodata patterns
                if nodata is not None:
                    # Exact nodata value
                    invalid = np.all(data == nodata, axis=2)
                    valid_mask &= ~invalid
                
                # Black pixels (0,0,0)
                black = np.all(data == 0, axis=2)
                valid_mask &= ~black
                
                # White pixels (255,255,255)
                white = np.all(data == 255, axis=2)
                valid_mask &= ~white
                
                if not valid_mask.any():
                    st.error("No valid data in RGB image")
                    return None, None
                
                # Normalize RGB to 0-1
                data_norm = data / 255.0
                data_norm = np.clip(data_norm, 0, 1)
                
                # Create RGBA
                rgba = np.zeros((out_h, out_w, 4), dtype=np.float32)
                rgba[:, :, :3] = data_norm
                rgba[:, :, 3] = valid_mask.astype(np.float32)  # Alpha channel
                
            else:
                # Single band (MBES, Mag, or grayscale SSS)
                
                # FIX 1: CRITICAL - Proper nodata masking
                valid_mask = np.ones(data.shape, dtype=bool)
                
                # Handle explicit nodata value
                if nodata is not None:
                    valid_mask &= (data != nodata)
                
                # Handle common SSS nodata patterns
                if is_sss:
                    # SSS often has 0 as nodata
                    valid_mask &= (data != 0)
                    # Very high values (sensor saturation)
                    valid_mask &= (data < 255)
                
                # Handle NaN/Inf
                valid_mask &= np.isfinite(data)
                
                if not valid_mask.any():
                    st.error("No valid data found")
                    return None, None
                
                valid_data = data[valid_mask]
                
                # FIX 1: CRITICAL - Aggressive contrast stretching for SSS
                if is_sss:
                    # SSS needs tight percentiles for good contrast
                    vmin, vmax = np.percentile(valid_data, [0.5, 99.5])
                elif is_mag:
                    # Mag: Symmetric around zero
                    abs_max = np.percentile(np.abs(valid_data), 99)
                    vmin, vmax = -abs_max, abs_max
                else:
                    # MBES: Standard percentiles
                    vmin, vmax = np.percentile(valid_data, [2, 98])
                
                # Avoid divide by zero
                if vmax == vmin:
                    vmax = vmin + 1
                
                # Normalize to 0-1
                data_norm = (data - vmin) / (vmax - vmin)
                data_norm = np.clip(data_norm, 0, 1)
                
                # FIX 1: CRITICAL - Set invalid to 0, not NaN
                data_norm[~valid_mask] = 0
                
                # Apply colormap
                cmap = get_cmap(colormap)
                rgba = cmap(data_norm)
                
                # FIX 1: CRITICAL - Alpha channel for transparency
                rgba[:, :, 3] = valid_mask.astype(np.float32)
            
            # Convert to uint8
            rgba_uint8 = (rgba * 255).astype(np.uint8)
            
            # Create PIL image
            img = Image.fromarray(rgba_uint8, mode='RGBA')
            
            # Encode to PNG
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            
            return img_base64, bounds_wgs84
            
    except Exception as e:
        st.error(f"Error processing raster: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
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
    
    # Add rasters with proper opacity
    for i, (img_base64, bounds) in enumerate(raster_layers):
        if img_base64 and bounds:
            img_url = f"data:image/png;base64,{img_base64}"
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.85,  # Slightly higher for better visibility
                interactive=True,
                name=f"Raster {i+1}"
            ).add_to(m)
    
    # Add vectors
    for gdf, color, name in vector_layers:
        if gdf is not None:
            gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs and str(gdf.crs) != 'EPSG:4326' else gdf
            folium.GeoJson(gdf_wgs84, name=name,
                         style_function=lambda x, c=color: {'color': c, 'weight': 2}).add_to(m)
    
    # Add mag points
    if mag_points:
        for point in mag_points:
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
    
    progress_container = st.container()
    
    with progress_container:
        st.info(f"📡 Loading {len(sss_id_list)} SSS tiles at {quality_preset}...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        
        for i, file_id in enumerate(sss_id_list):
            try:
                status_text.text(f"Processing tile {i+1}/{len(sss_id_list)}...")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                    download_from_gdrive(file_id, tmp.name)
                    
                    # CRITICAL: Mark as SSS for proper processing
                    img, bounds = tif_to_png_base64(
                        tmp.name,
                        colormap='gray',
                        max_size=max_pixels,
                        is_sss=True  # This triggers SSS-specific processing
                    )
                    
                    if img and bounds:
                        st.session_state.raster_layers.append((img, bounds))
                        success_count += 1
                    
                    os.unlink(tmp.name)
                
                progress_bar.progress((i + 1) / len(sss_id_list))
                
            except Exception as e:
                status_text.warning(f"Tile {i+1} failed: {e}")
        
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
                status_text.text(f"Processing MBES {i+1}/{len(mbes_id_list)}...")
                
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
        st.success(f"✅ Loaded MBES!")

# FIX 3: Magnetometer TIF with FILE UPLOAD (not just Google Drive)
st.sidebar.subheader("🧲 Magnetometer")

mag_upload_method = st.sidebar.radio(
    "Mag TIF Source:",
    ["Upload File", "Google Drive ID"],
    key="mag_method"
)

if mag_upload_method == "Upload File":
    # FIX 3: Direct file upload for ANY raster format
    mag_tif_file = st.sidebar.file_uploader(
        "Upload Mag TIF",
        type=['tif', 'tiff', 'img', 'geotiff'],
        help="Upload magnetometer GeoTIFF directly"
    )
    
    if mag_tif_file is not None and st.sidebar.button("Load Uploaded Mag TIF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            try:
                tmp.write(mag_tif_file.getvalue())
                tmp.flush()
                
                st.info("🧲 Processing magnetometer data...")
                
                img, bounds = tif_to_png_base64(
                    tmp.name,
                    colormap='seismic',  # Red-white-blue for mag
                    max_size=max_pixels,
                    is_sss=False,
                    is_mag=True
                )
                
                if img and bounds:
                    st.session_state.raster_layers.append((img, bounds))
                    st.success("✅ Mag TIF loaded!")
                else:
                    st.error("❌ Failed to process mag TIF")
                
                os.unlink(tmp.name)
                
            except Exception as e:
                st.error(f"Error loading mag TIF: {e}")
                import traceback
                st.code(traceback.format_exc())

else:
    # Google Drive method
    mag_tif_ids = st.sidebar.text_area(
        "Mag TIF File IDs",
        height=60,
        help="Gridded magnetometer data from Google Drive"
    )
    
    if mag_tif_ids and st.sidebar.button("Load Mag from Drive"):
        mag_id_list = [fid.strip() for fid in mag_tif_ids.strip().split('\n') if fid.strip()]
        
        for file_id in mag_id_list:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                try:
                    download_from_gdrive(file_id, tmp.name)
                    
                    img, bounds = tif_to_png_base64(
                        tmp.name,
                        colormap='seismic',
                        max_size=max_pixels,
                        is_sss=False,
                        is_mag=True
                    )
                    
                    if img and bounds:
                        st.session_state.raster_layers.append((img, bounds))
                    
                    os.unlink(tmp.name)
                except Exception as e:
                    st.error(f"Failed: {e}")
        
        st.success("✅ Mag TIF loaded!")

# Mag CSV targets
st.sidebar.subheader("🎯 Mag Targets (CSV)")

mag_csv_file = st.sidebar.file_uploader(
    "Upload Mag Targets CSV",
    type=['csv'],
    help="CSV with columns: lat, lon, magnitude"
)

if mag_csv_file is not None:
    try:
        df = pd.read_csv(mag_csv_file)
        
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
    
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Vectors
st.sidebar.subheader("📏 Vectors")

vector_id = st.sidebar.text_input("Vector File ID (GeoJSON)")
if vector_id and st.sidebar.button("Load Vector"):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
        try:
            download_from_gdrive(vector_id, tmp.name)
            gdf = gpd.read_file(tmp.name)
            st.session_state.vector_layers.append((gdf, 'yellow', 'Vector'))
            st.success(f"✅ Vector added")
            os.unlink(tmp.name)
        except Exception as e:
            st.error(f"Error: {e}")

# Clear
if st.sidebar.button("🗑️ Clear All"):
    st.session_state.raster_layers = []
    st.session_state.vector_layers = []
    st.session_state.mag_points = []
    st.success("Cleared!")
    st.rerun()

# Display
if st.session_state.raster_layers or st.session_state.vector_layers or st.session_state.mag_points:
    st.subheader("🗺️ Survey Map")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rasters", len(st.session_state.raster_layers))
    with col2:
        st.metric("Vectors", len(st.session_state.vector_layers))
    with col3:
        st.metric("Mag Targets", len(st.session_state.mag_points))
    with col4:
        st.metric("Quality", quality_preset.split('(')[0])
    
    m = create_map(st.session_state.raster_layers, st.session_state.vector_layers,
                  st.session_state.mag_points, basemap)
    st_folium(m, width=1400, height=800, key="main_map")
else:
    st.info("👆 Load data using sidebar")
    
    st.markdown("""
    ## 🗺️ Marine Survey Viewer
    
    ### Quality Settings:
    - **Fast (500px)**: Quick loading, good for 44 tiles
    - **Good (1000px)**: Balanced quality/speed
    - **High (2000px)**: High detail, 10-20 tiles max
    - **Ultra (3000px)**: Very high detail, 5-10 tiles
    - **Full Resolution**: Maximum detail, 1-5 tiles only
    
    ### SSS Black Box Fix Applied:
    - Proper nodata detection
    - Aggressive contrast stretching
    - Transparent background
    
    ### Mag TIF Support:
    - Upload directly OR use Google Drive
    - All GeoTIFF formats supported
    - Red-white-blue colormap
    """)
