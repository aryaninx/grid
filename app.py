#!/usr/bin/env python3
"""
THE GRID - Marine Survey Hazard Visualization Platform
Demo version showcasing multi-sensor hazard detection visualization
"""

import streamlit as st
import rasterio
from rasterio.warp import transform_bounds
import geopandas as gpd
import numpy as np
from PIL import Image
import folium
from folium import plugins
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
from datetime import datetime

st.set_page_config(page_title="The Grid - Marine Risk Viewer", layout="wide", page_icon="⚠️")

# Session state
if 'raster_layers' not in st.session_state:
    st.session_state.raster_layers = []
if 'vector_layers' not in st.session_state:
    st.session_state.vector_layers = []
if 'hazards' not in st.session_state:
    st.session_state.hazards = []
if 'turbines' not in st.session_state:
    st.session_state.turbines = []
if 'sbp_lines' not in st.session_state:
    st.session_state.sbp_lines = []
if 'mag_tif_layer' not in st.session_state:
    st.session_state.mag_tif_layer = None

st.title("⚠️ The Grid - Marine Hazard Visualization")

# Sidebar
st.sidebar.header("📁 Data Loading")

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

st.sidebar.header("🎨 Display Settings")
basemap = st.sidebar.selectbox("Basemap", ['OpenStreetMap', 'Esri Satellite'], index=0)
mbes_colormap = st.sidebar.selectbox("MBES/Mag Colormap", ['ocean', 'viridis', 'seismic'], index=0)


def download_from_gdrive(file_id, output_path):
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
    try:
        with rasterio.open(file_path) as src:
            orig_h, orig_w = src.height, src.width
            if max_size >= 10000:
                downsample = 1
                out_h, out_w = orig_h, orig_w
            else:
                downsample = max(1, int(max(orig_h, orig_w) / max_size))
                out_h = orig_h // downsample
                out_w = orig_w // downsample
            
            if src.count >= 3:
                data = np.zeros((out_h, out_w, 3), dtype=np.float32)
                for i in range(3):
                    band = src.read(i + 1, out_shape=(out_h, out_w),
                                   resampling=rasterio.enums.Resampling.average)
                    data[:, :, i] = band.astype(np.float32)
                is_rgb = True
            else:
                data = src.read(1, out_shape=(out_h, out_w),
                              resampling=rasterio.enums.Resampling.average)
                data = data.astype(np.float32)
                is_rgb = False
            
            try:
                bounds_wgs84 = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            except:
                bounds_wgs84 = src.bounds
            
            nodata = src.nodata
            
            if is_rgb:
                valid_mask = np.ones((out_h, out_w), dtype=bool)
                if nodata is not None:
                    invalid = np.all(data == nodata, axis=2)
                    valid_mask &= ~invalid
                black = np.all(data == 0, axis=2)
                valid_mask &= ~black
                white = np.all(data == 255, axis=2)
                valid_mask &= ~white
                
                if not valid_mask.any():
                    return None, None
                
                data_norm = np.clip(data / 255.0, 0, 1)
                rgba = np.zeros((out_h, out_w, 4), dtype=np.float32)
                rgba[:, :, :3] = data_norm
                rgba[:, :, 3] = valid_mask.astype(np.float32)
            else:
                valid_mask = np.ones(data.shape, dtype=bool)
                if nodata is not None:
                    valid_mask &= (data != nodata)
                if is_sss:
                    valid_mask &= (data != 0)
                    valid_mask &= (data < 255)
                valid_mask &= np.isfinite(data)
                
                if not valid_mask.any():
                    return None, None
                
                valid_data = data[valid_mask]
                
                if is_sss:
                    vmin, vmax = np.percentile(valid_data, [0.5, 99.5])
                elif is_mag:
                    abs_max = np.percentile(np.abs(valid_data), 99)
                    vmin, vmax = -abs_max, abs_max
                else:
                    vmin, vmax = np.percentile(valid_data, [2, 98])
                
                if vmax == vmin:
                    vmax = vmin + 1
                
                data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
                data_norm[~valid_mask] = 0
                
                cmap = get_cmap(colormap)
                rgba = cmap(data_norm)
                rgba[:, :, 3] = valid_mask.astype(np.float32)
            
            rgba_uint8 = (rgba * 255).astype(np.uint8)
            img = Image.fromarray(rgba_uint8, mode='RGBA')
            
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            
            return img_base64, bounds_wgs84
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None


def get_hazard_icon(hazard_type):
    icons = {
        'Wreck': 'ship',
        'UXO': 'exclamation-triangle',
        'Boulder': 'circle',
        'Boulder Field': 'circle',
        'Shallow Gas': 'cloud',
        'Buried Channel': 'water',
        'Sand Wave Field': 'water',
        'Hard Ground': 'square',
        'Pipeline': 'minus'
    }
    return icons.get(hazard_type, 'info-sign')


def get_risk_color(risk):
    colors = {
        'Critical': '#DC143C',
        'High': '#FF8C00',
        'Medium': '#FFD700',
        'Low': '#90EE90'
    }
    return colors.get(risk, '#808080')


def create_map(raster_layers, vector_layers, hazards, turbines, sbp_lines, mag_tif, basemap_choice,
               show_sbp=True, show_mag_tif=True):
    
    all_bounds = [b for _, b in raster_layers if b]
    if all_bounds:
        min_lon = min(b[0] for b in all_bounds)
        min_lat = min(b[1] for b in all_bounds)
        max_lon = max(b[2] for b in all_bounds)
        max_lat = max(b[3] for b in all_bounds)
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
    else:
        center_lat, center_lon = 53.77, 0.15
    
    if basemap_choice == 'Esri Satellite':
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
        folium.TileLayer(tiles=tiles, attr='Esri').add_to(m)
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')
    
    # Add rasters (SSS/MBES)
    for i, (img_base64, bounds) in enumerate(raster_layers):
        if img_base64 and bounds:
            img_url = f"data:image/png;base64,{img_base64}"
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.85,
                interactive=True,
                name=f"Survey Data {i+1}"
            ).add_to(m)
    
    # Add Mag TIF (if loaded and toggle on)
    if show_mag_tif and mag_tif:
        img_base64, bounds = mag_tif
        if img_base64 and bounds:
            img_url = f"data:image/png;base64,{img_base64}"
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.6,
                interactive=True,
                name="Magnetometer TIF"
            ).add_to(m)
    
    # Add vectors
    for gdf, color, name in vector_layers:
        if gdf is not None:
            try:
                gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs and str(gdf.crs) != 'EPSG:4326' else gdf
                folium.GeoJson(gdf_wgs84, name=name,
                             style_function=lambda x, c=color: {'color': c, 'weight': 2}).add_to(m)
            except:
                pass
    
    # Add SBP lines (if toggle on)
    if show_sbp and sbp_lines is not None and len(sbp_lines) > 0:
        try:
            gdf_sbp = sbp_lines.to_crs('EPSG:4326') if sbp_lines.crs else sbp_lines
            
            # Create simple GeoJSON without complex properties for tooltips
            sbp_geojson = {
                "type": "FeatureCollection",
                "features": []
            }
            
            for idx, row in gdf_sbp.iterrows():
                # Get line name safely
                line_name = str(row.get('line_name', row.get('name', row.get('id', f'Line {idx+1}'))))
                
                feature = {
                    "type": "Feature",
                    "geometry": row.geometry.__geo_interface__,
                    "properties": {
                        "name": line_name  # Only simple string property
                    }
                }
                sbp_geojson["features"].append(feature)
            
            folium.GeoJson(
                sbp_geojson,
                name="SBP Survey Lines",
                style_function=lambda x: {
                    'color': '#00FFFF',
                    'weight': 2,
                    'opacity': 0.7
                },
                tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Line:'])
            ).add_to(m)
        except Exception as e:
            st.warning(f"Could not display SBP lines: {e}")
    
    # Add turbines
    if turbines is not None and len(turbines) > 0:
        try:
            gdf_turb = turbines.to_crs('EPSG:4326') if turbines.crs else turbines
            for idx, row in gdf_turb.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=8,
                    color='#4169E1',
                    fill=True,
                    fillColor='#4169E1',
                    fillOpacity=0.8,
                    popup=f"<b>{row['name']}</b><br>Status: {row.get('status', 'Unknown')}",
                    tooltip=row['name']
                ).add_to(m)
        except:
            pass
    
    # Add hazards with detailed popups
    if hazards is not None and len(hazards) > 0:
        try:
            gdf_haz = hazards.to_crs('EPSG:4326') if hazards.crs else hazards
            for idx, row in gdf_haz.iterrows():
                risk = row.get('risk', 'Unknown')
                hazard_type = row.get('hazard_type', 'Unknown')
                
                # Build detailed popup
                sensors_list = ', '.join(row.get('detected_by', []))
                
                popup_html = f"""
                <div style="width:320px; font-family:Arial;">
                    <h3 style="margin:0; color:{get_risk_color(risk)}; border-bottom:2px solid {get_risk_color(risk)}; padding-bottom:5px;">
                        {hazard_type} - {row.get('id', '')}
                    </h3>
                    
                    <div style="margin-top:10px;">
                        <p style="margin:5px 0;"><b>Name:</b> {row.get('name', 'N/A')}</p>
                        <p style="margin:5px 0;"><b>Size:</b> {row.get('size', 'N/A')}</p>
                    </div>
                    
                    <div style="margin-top:10px; background:#f0f0f0; padding:8px; border-radius:4px;">
                        <h4 style="margin:5px 0; color:#333;">📡 Detected By:</h4>
                        <p style="margin:5px 0;">{sensors_list}</p>
                    </div>
                    
                    <div style="margin-top:10px;">
                        <p style="margin:5px 0;"><b>Distance to Turbine:</b> {row.get('distance_to_turbine_m', 'N/A')}m</p>
                        <p style="margin:5px 0;"><b>Nearest:</b> {row.get('nearest_turbine', 'N/A')}</p>
                    </div>
                    
                    <div style="margin-top:10px; background:#fff3cd; padding:8px; border-radius:4px;">
                        <h4 style="margin:5px 0; color:#333;">⚠️ Risk Assessment</h4>
                        <p style="margin:5px 0;"><b>Level:</b> <span style="color:{get_risk_color(risk)}; font-weight:bold;">{risk}</span></p>
                        <p style="margin:5px 0;"><b>Score:</b> {row.get('risk_score', 'N/A')}/10</p>
                    </div>
                    
                    <div style="margin-top:10px; background:#f0f0f0; padding:8px; border-radius:4px;">
                        <h4 style="margin:5px 0; color:#333;">🎯 Action Required</h4>
                        <p style="margin:5px 0;">{row.get('action', 'N/A')}</p>
                    </div>
                    
                    <div style="margin-top:10px; background:#d4edda; padding:8px; border-radius:4px;">
                        <h4 style="margin:5px 0; color:#333;">💰 Estimated Cost</h4>
                        <p style="margin:5px 0; font-weight:bold;">{row.get('cost', 'N/A')}</p>
                    </div>
                    
                    <div style="margin-top:10px; background:#d1ecf1; padding:8px; border-radius:4px;">
                        <h4 style="margin:5px 0; color:#333;">📅 Timeline</h4>
                        <p style="margin:5px 0;"><b>Investigation:</b> {row.get('investigation_timeline', 'N/A')}</p>
                        <p style="margin:5px 0;"><b>Expected Failure:</b> {row.get('expected_failure', 'N/A')}</p>
                    </div>
                    
                    <div style="margin-top:10px; font-size:11px; color:#666; border-top:1px solid #ddd; padding-top:5px;">
                        <p style="margin:2px 0;"><b>Details:</b> {row.get('details', 'N/A')}</p>
                    </div>
                </div>
                """
                
                # Tooltip (hover)
                tooltip_text = f"""<b>{hazard_type}</b><br>{row.get('name', '')}<br>Risk: {risk}<br>{row.get('distance_to_turbine_m', 'N/A')}m to {row.get('nearest_turbine', '')}"""
                
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=folium.Popup(popup_html, max_width=340),
                    tooltip=folium.Tooltip(tooltip_text),
                    icon=folium.Icon(
                        color='red' if risk == 'Critical' else 'orange' if risk == 'High' else 'beige',
                        icon=get_hazard_icon(hazard_type),
                        prefix='fa'
                    )
                ).add_to(m)
        except Exception as e:
            st.warning(f"Error adding hazards to map: {e}")
    
    folium.plugins.Fullscreen().add_to(m)
    folium.plugins.MousePosition().add_to(m)
    folium.plugins.MeasureControl(position='topleft').add_to(m)
    folium.LayerControl().add_to(m)
    
    return m


# ==============================================================================
# DATA LOADING
# ==============================================================================

st.sidebar.subheader("📡 SSS Data")
sss_ids = st.sidebar.text_area("SSS File IDs", height=80)

if sss_ids and st.sidebar.button("🚀 Load SSS"):
    sss_id_list = [fid.strip() for fid in sss_ids.strip().split('\n') if fid.strip()]
    with st.container():
        st.info(f"Loading {len(sss_id_list)} SSS tiles...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        success_count = 0
        for i, file_id in enumerate(sss_id_list):
            try:
                status_text.text(f"Tile {i+1}/{len(sss_id_list)}...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                    download_from_gdrive(file_id, tmp.name)
                    img, bounds = tif_to_png_base64(tmp.name, colormap='gray',
                                                   max_size=max_pixels, is_sss=True)
                    if img and bounds:
                        st.session_state.raster_layers.append((img, bounds))
                        success_count += 1
                    os.unlink(tmp.name)
                progress_bar.progress((i + 1) / len(sss_id_list))
            except:
                pass
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ {success_count}/{len(sss_id_list)} SSS tiles loaded!")

# MBES
st.sidebar.subheader("🗺️ MBES Data")
mbes_ids = st.sidebar.text_area("MBES File IDs", height=40)

if mbes_ids and st.sidebar.button("Load MBES"):
    for file_id in [fid.strip() for fid in mbes_ids.strip().split('\n') if fid.strip()]:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            try:
                download_from_gdrive(file_id, tmp.name)
                img, bounds = tif_to_png_base64(tmp.name, colormap=mbes_colormap, max_size=max_pixels)
                if img and bounds:
                    st.session_state.raster_layers.append((img, bounds))
                os.unlink(tmp.name)
            except:
                pass
    st.success("✅ MBES loaded!")

# Mag TIF
st.sidebar.subheader("🧲 Magnetometer TIF")
mag_tif_id = st.sidebar.text_input("Mag TIF File ID")

if mag_tif_id and st.sidebar.button("Load Mag TIF"):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        try:
            st.info("Loading magnetometer data...")
            download_from_gdrive(mag_tif_id, tmp.name)
            img, bounds = tif_to_png_base64(tmp.name, colormap='seismic',
                                           max_size=max_pixels, is_mag=True)
            if img and bounds:
                st.session_state.mag_tif_layer = (img, bounds)
                st.success("✅ Mag TIF loaded!")
            os.unlink(tmp.name)
        except Exception as e:
            st.error(f"Failed: {e}")

# SBP Lines
st.sidebar.subheader("🔊 SBP Survey Lines")
sbp_file = st.sidebar.file_uploader("Upload SBP Lines (GeoJSON)", type=['geojson', 'json'])

if sbp_file:
    try:
        st.session_state.sbp_lines = gpd.read_file(sbp_file)
        st.sidebar.success(f"✅ Loaded {len(st.session_state.sbp_lines)} SBP lines")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Turbines
st.sidebar.subheader("🔌 Turbines")

# Option 1: Google Drive
turbine_ids = st.sidebar.text_area("Turbine GeoJSON File IDs", height=40, 
                                   help="Enter Google Drive file IDs, one per line")
if turbine_ids and st.sidebar.button("Load Turbines (GDrive)"):
    turbine_id_list = [fid.strip() for fid in turbine_ids.strip().split('\n') if fid.strip()]
    for file_id in turbine_id_list:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
            try:
                download_from_gdrive(file_id, tmp.name)
                st.session_state.turbines = gpd.read_file(tmp.name)
                st.sidebar.success(f"✅ Loaded {len(st.session_state.turbines)} turbines")
                os.unlink(tmp.name)
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

# Option 2: File upload
turbine_file = st.sidebar.file_uploader("OR Upload Turbines (GeoJSON)", type=['geojson', 'json'])

if turbine_file:
    try:
        st.session_state.turbines = gpd.read_file(turbine_file)
        st.sidebar.success(f"✅ Loaded {len(st.session_state.turbines)} turbines")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Hazards
st.sidebar.subheader("⚠️ Hazards")
hazard_file = st.sidebar.file_uploader("Upload Hazards (GeoJSON)", type=['geojson', 'json'])

if hazard_file:
    try:
        st.session_state.hazards = gpd.read_file(hazard_file)
        st.sidebar.success(f"✅ Loaded {len(st.session_state.hazards)} hazards")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if st.sidebar.button("🗑️ Clear All"):
    st.session_state.raster_layers = []
    st.session_state.vector_layers = []
    st.session_state.hazards = []
    st.session_state.turbines = []
    st.session_state.sbp_lines = []
    st.session_state.mag_tif_layer = None
    st.success("Cleared!")
    st.rerun()

# ==============================================================================
# HAZARD FILTERS
# ==============================================================================

# Display map if ANY data is loaded
has_data = (
    len(st.session_state.raster_layers) > 0 or
    len(st.session_state.vector_layers) > 0 or
    (st.session_state.hazards is not None and len(st.session_state.hazards) > 0) or
    (st.session_state.turbines is not None and len(st.session_state.turbines) > 0) or
    (st.session_state.sbp_lines is not None and len(st.session_state.sbp_lines) > 0) or
    st.session_state.mag_tif_layer is not None
)

if has_data:
    # Only show filters if we have hazards
    if st.session_state.hazards is not None and len(st.session_state.hazards) > 0:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Hazard Filters")
        
        # Get unique hazard types
        hazard_types = sorted(st.session_state.hazards['hazard_type'].unique().tolist())
        
        selected_hazards = st.sidebar.multiselect(
            "Show Hazard Types:",
            hazard_types,
            default=hazard_types,
            help="Select which hazard types to display"
        )
        
        # Risk filter
        selected_risks = st.sidebar.multiselect(
            "Show Risk Levels:",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"]
        )
        
        # Timeline filter
        st.sidebar.markdown("### 📅 Investigation Timeline")
        
        selected_timelines = st.sidebar.multiselect(
            "Show by Timeline:",
            ["3 months", "6 months", "1 year", "5 years", "All"],
            default=["All"]
        )
        
        # Expected failure filter
        st.sidebar.markdown("### ⏰ Expected Time to Failure")
        
        selected_failures = st.sidebar.multiselect(
            "Show by Failure Timeline:",
            ["3 months", "6 months", "N/A", "All"],
            default=["All"]
        )
        
        # Filter hazards
        filtered_hazards = st.session_state.hazards.copy()
        
        # Apply filters
        if selected_hazards:
            filtered_hazards = filtered_hazards[filtered_hazards['hazard_type'].isin(selected_hazards)]
        
        if selected_risks:
            filtered_hazards = filtered_hazards[filtered_hazards['risk'].isin(selected_risks)]
        
        if "All" not in selected_timelines:
            filtered_hazards = filtered_hazards[filtered_hazards['investigation_timeline'].isin(selected_timelines)]
        
        if "All" not in selected_failures:
            filtered_hazards = filtered_hazards[filtered_hazards['expected_failure'].isin(selected_failures)]
    else:
        # No hazards loaded, use empty GeoDataFrame
        filtered_hazards = gpd.GeoDataFrame()
    
    # Toggle layers (show always if any data loaded)
    st.sidebar.markdown("---")
    st.sidebar.subheader("👁️ Layer Toggles")
    show_sbp = st.sidebar.checkbox("Show SBP Lines", value=True)
    show_mag_tif = st.sidebar.checkbox("Show Magnetometer TIF", value=True)
    
    
    # Summary metrics (only show if hazards exist)
    if len(filtered_hazards) > 0:
        st.subheader("📊 Hazard Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            critical = len(filtered_hazards[filtered_hazards['risk'] == 'Critical'])
            st.metric("🔴 Critical", critical)
        
        with col2:
            high = len(filtered_hazards[filtered_hazards['risk'] == 'High'])
            st.metric("🟠 High", high)
        
        with col3:
            medium = len(filtered_hazards[filtered_hazards['risk'] == 'Medium'])
            st.metric("🟡 Medium", medium)
        
        with col4:
            low = len(filtered_hazards[filtered_hazards['risk'] == 'Low'])
            st.metric("🟢 Low", low)
        
        with col5:
            st.metric("📍 Total Shown", len(filtered_hazards))
    
    # Map (show always)
    st.subheader("🗺️ Interactive Map")
    if len(filtered_hazards) > 0:
        st.info("💡 **Click** markers for detailed info | **Hover** for quick summary")
    
    m = create_map(
        st.session_state.raster_layers,
        st.session_state.vector_layers,
        filtered_hazards,
        st.session_state.turbines,
        st.session_state.sbp_lines,
        st.session_state.mag_tif_layer,
        basemap,
        show_sbp=show_sbp,
        show_mag_tif=show_mag_tif
    )
    st_folium(m, width=1400, height=700)
    
    # Hazard table
    if len(filtered_hazards) > 0:
        st.subheader("📋 Hazard Register")
        
        display_cols = ['id', 'hazard_type', 'risk', 'distance_to_turbine_m', 
                       'investigation_timeline', 'expected_failure', 'action', 'cost']
        
        df_display = filtered_hazards[display_cols].copy()
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Export
        csv = filtered_hazards.to_csv(index=False)
        st.download_button(
            "📥 Download Hazard Register (CSV)",
            csv,
            f"hazard_register_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
        
        # Breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### By Hazard Type")
            type_counts = filtered_hazards['hazard_type'].value_counts()
            for htype, count in type_counts.items():
                st.write(f"• **{htype}**: {count}")
        
        with col2:
            st.markdown("### By Investigation Timeline")
            timeline_counts = filtered_hazards['investigation_timeline'].value_counts()
            for timeline, count in timeline_counts.items():
                st.write(f"• **{timeline}**: {count} hazards")

else:
    st.info("👆 Upload hazard GeoJSON file to begin visualization")
    
    st.markdown("""
    ## 🎯 The Grid - Demo Instructions
    
    ### Step 1: Load Survey Data
    - **SSS tiles**: Background imagery
    - **MBES**: Bathymetry
    - **Mag TIF**: Magnetometer heatmap
    
    ### Step 2: Load Infrastructure
    - **Turbines GeoJSON**: Wind turbine locations
    - **SBP Lines GeoJSON**: Survey line coverage
    
    ### Step 3: Load Hazards
    - **Hazards GeoJSON**: All detected hazards
    - Filter by type, risk, timeline
    - Click markers for detailed popups
    
    ### Demo Files Available:
    - `demo_hazards_all.geojson` - 14 sample hazards
    - `demo_turbines.geojson` - 9 turbine locations  
    - `demo_sbp_lines.geojson` - 6 survey lines
    """)
