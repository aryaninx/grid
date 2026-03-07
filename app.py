#!/usr/bin/env python3
"""
THE GRID - DEBUG MODE
Comprehensive error logging and testing
"""

import streamlit as st
import traceback

# Debug log
if 'debug_log' not in st.session_state:
    st.session_state.debug_log = []

def log_debug(message, error=None):
    log_entry = f"[{len(st.session_state.debug_log)}] {message}"
    if error:
        log_entry += f"\nError: {str(error)}\n{traceback.format_exc()}"
    st.session_state.debug_log.append(log_entry)

try:
    log_debug("Importing modules...")
    import rasterio
    from rasterio.warp import transform_bounds
    import geopandas as gpd
    import numpy as np
    from PIL import Image
    import folium
    from streamlit_folium import st_folium
    import tempfile, os, base64, requests
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.cm import get_cmap
    log_debug("✅ Imports successful")
except Exception as e:
    log_debug("❌ Import failed", e)
    st.error(f"Import error: {e}\n{traceback.format_exc()}")
    st.stop()

st.set_page_config(page_title="The Grid - DEBUG", layout="wide", page_icon="🐛")

# Session state
for key in ['raster_layers', 'vector_layers', 'hazards', 'turbines', 'sbp_lines', 'mag_tif_layer']:
    if key not in st.session_state:
        st.session_state[key] = [] if key != 'mag_tif_layer' else None

st.title("🐛 The Grid - DEBUG MODE")

# Debug Log
with st.expander("🔍 Debug Log (Click to expand)", expanded=True):
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Clear Log"):
            st.session_state.debug_log = []
            st.rerun()
    with col2:
        st.caption(f"{len(st.session_state.debug_log)} log entries")
    
    if st.session_state.debug_log:
        st.code("\n\n".join(st.session_state.debug_log[-20:]), language="text")
    else:
        st.info("No debug messages yet. Upload files to see logs.")

# What's the current error?
st.markdown("---")
st.header("❓ What error are you seeing?")
st.info("**Common errors:**\n- AssertionError when uploading GeoJSON → Usually from list in properties\n- ValueError with GeoDataFrame → Boolean check issue\n- Import errors → Missing packages\n- Download fails → Invalid Google Drive File ID")

# Test specific functionality
st.sidebar.header("🧪 Test Specific Features")

test_option = st.sidebar.selectbox("Test what?", [
    "Nothing",
    "Upload Hazards GeoJSON", 
    "Upload SBP Lines GeoJSON",
    "Test Google Drive Download",
    "Test TIF Processing"
])

if test_option == "Upload Hazards GeoJSON":
    st.subheader("Test: Upload Hazards")
    uploaded = st.file_uploader("Upload hazards GeoJSON", type=['geojson', 'json'])
    
    if uploaded:
        try:
            log_debug(f"Reading uploaded file: {uploaded.name}")
            st.session_state.hazards = gpd.read_file(uploaded)
            log_debug(f"✅ Loaded {len(st.session_state.hazards)} features")
            
            st.success(f"✅ Loaded {len(st.session_state.hazards)} hazards")
            st.dataframe(st.session_state.hazards.head())
            
            # Check for problematic properties
            st.subheader("Property Check")
            for col in st.session_state.hazards.columns:
                sample = st.session_state.hazards[col].iloc[0]
                type_str = str(type(sample))
                if isinstance(sample, list):
                    st.warning(f"⚠️ Column '{col}' contains lists: {sample}")
                elif isinstance(sample, dict):
                    st.warning(f"⚠️ Column '{col}' contains dicts: {sample}")
                else:
                    st.success(f"✅ Column '{col}': {type_str}")
                    
        except Exception as e:
            st.error(f"❌ Failed to load: {e}")
            log_debug("Hazards upload failed", e)
            st.code(traceback.format_exc())

elif test_option == "Upload SBP Lines GeoJSON":
    st.subheader("Test: Upload SBP Lines")
    uploaded = st.file_uploader("Upload SBP lines GeoJSON", type=['geojson', 'json'])
    
    if uploaded:
        try:
            log_debug(f"Reading uploaded file: {uploaded.name}")
            st.session_state.sbp_lines = gpd.read_file(uploaded)
            log_debug(f"✅ Loaded {len(st.session_state.sbp_lines)} features")
            
            st.success(f"✅ Loaded {len(st.session_state.sbp_lines)} SBP lines")
            st.dataframe(st.session_state.sbp_lines.head())
            
            # Check properties
            st.subheader("Property Check")
            for col in st.session_state.sbp_lines.columns:
                if col != 'geometry':
                    sample = st.session_state.sbp_lines[col].iloc[0]
                    type_str = str(type(sample))
                    if isinstance(sample, (list, dict)):
                        st.warning(f"⚠️ Column '{col}' contains complex type: {type_str}")
                    else:
                        st.success(f"✅ Column '{col}': {type_str}")
                        
        except Exception as e:
            st.error(f"❌ Failed: {e}")
            log_debug("SBP upload failed", e)
            st.code(traceback.format_exc())

elif test_option == "Test Google Drive Download":
    st.subheader("Test: Google Drive Download")
    test_id = st.text_input("Enter Google Drive File ID")
    
    if st.button("Test Download") and test_id:
        try:
            log_debug(f"Testing download of {test_id}")
            url = f"https://drive.google.com/uc?id={test_id}&export=download"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.test') as tmp:
                session = requests.Session()
                response = session.get(url, stream=True, timeout=30)
                
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        params = {'id': test_id, 'confirm': value}
                        response = session.get(url, params=params, stream=True, timeout=30)
                
                with open(tmp.name, 'wb') as f:
                    f.write(response.content)
                
                size = os.path.getsize(tmp.name)
                log_debug(f"✅ Downloaded {size} bytes")
                st.success(f"✅ Downloaded {size:,} bytes")
                os.unlink(tmp.name)
                
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            log_debug("GDrive download test failed", e)
            st.code(traceback.format_exc())

elif test_option == "Test TIF Processing":
    st.subheader("Test: TIF Processing")
    uploaded_tif = st.file_uploader("Upload TIF file", type=['tif', 'tiff'])
    
    if uploaded_tif:
        try:
            log_debug(f"Processing uploaded TIF: {uploaded_tif.name}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                tmp.write(uploaded_tif.read())
                tmp_path = tmp.name
            
            log_debug(f"Temp file: {tmp_path}")
            
            # Try to open with rasterio
            with rasterio.open(tmp_path) as src:
                st.success(f"✅ Opened with rasterio")
                st.write(f"Size: {src.width}x{src.height}")
                st.write(f"Bands: {src.count}")
                st.write(f"CRS: {src.crs}")
                st.write(f"Bounds: {src.bounds}")
                log_debug(f"TIF info: {src.width}x{src.height}, {src.count} bands, CRS={src.crs}")
            
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"❌ Failed: {e}")
            log_debug("TIF test failed", e)
            st.code(traceback.format_exc())

# Current state
st.sidebar.markdown("---")
st.sidebar.header("📊 Current State")
st.sidebar.metric("Raster layers", len(st.session_state.raster_layers))
st.sidebar.metric("Hazards", len(st.session_state.hazards) if isinstance(st.session_state.hazards, gpd.GeoDataFrame) else 0)
st.sidebar.metric("SBP lines", len(st.session_state.sbp_lines) if isinstance(st.session_state.sbp_lines, gpd.GeoDataFrame) else 0)
st.sidebar.metric("Turbines", len(st.session_state.turbines) if isinstance(st.session_state.turbines, gpd.GeoDataFrame) else 0)

# System info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ System Info")
st.sidebar.code(f"""Streamlit: {st.__version__}
Rasterio: {rasterio.__version__}
GeoPandas: {gpd.__version__}
NumPy: {np.__version__}""")

log_debug("Debug app ready")
