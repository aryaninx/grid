#!/usr/bin/env python3
"""
THE GRID - Marine Survey Hazard Visualization Platform
ONE-CLICK LOADING VERSION - All Google Drive File IDs pre-configured
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

st.set_page_config(page_title="The Grid", layout="wide", page_icon="⚠️")

# Session state
if 'raster_layers' not in st.session_state:
    st.session_state.raster_layers = []
if 'hazards' not in st.session_state:
    st.session_state.hazards = None
if 'turbines' not in st.session_state:
    st.session_state.turbines = None
if 'sbp_lines' not in st.session_state:
    st.session_state.sbp_lines = None
if 'mag_tif_layer' not in st.session_state:
    st.session_state.mag_tif_layer = None

st.title("⚠️ The Grid - Marine Hazard Visualization")

# ==============================================================================
# FUNCTIONS (MUST BE DEFINED FIRST!)
# ==============================================================================

def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive"""
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
def tif_to_png_base64(file_path, colormap='gray', max_size=1000, is_sss=False, is_mag=False):
    """Convert TIF to base64 PNG"""
    try:
        with rasterio.open(file_path) as src:
            orig_h, orig_w = src.height, src.width
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
        st.error(f"Error processing TIF: {str(e)}")
        return None, None

def get_risk_color(risk):
    """Get color for risk level"""
    colors = {
        'Critical': '#DC143C',
        'High': '#FF8C00',
        'Medium': '#FFD700',
        'Low': '#90EE90'
    }
    return colors.get(risk, '#808080')

def create_map(raster_layers, hazards, turbines, sbp_lines, mag_tif, basemap_choice, show_sbp=True, show_mag=True):
    """Create Folium map with all layers"""
    
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
        center_lat, center_lon = 53.81, 0.13
    
    # Create base map
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
                name=f"Survey Data {i+1}"
            ).add_to(m)
    
    # Add Mag TIF
    if show_mag and mag_tif:
        img_base64, bounds = mag_tif
        if img_base64 and bounds:
            img_url = f"data:image/png;base64,{img_base64}"
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.6,
                name="Magnetometer"
            ).add_to(m)
    
    # Add SBP lines
    if show_sbp and sbp_lines is not None and len(sbp_lines) > 0:
        try:
            gdf_sbp = sbp_lines.to_crs('EPSG:4326') if sbp_lines.crs else sbp_lines
            sbp_geojson = {"type": "FeatureCollection", "features": []}
            
            for idx, row in gdf_sbp.iterrows():
                line_name = str(row.get('file', row.get('name', f'Line {idx+1}')))
                feature = {
                    "type": "Feature",
                    "geometry": row.geometry.__geo_interface__,
                    "properties": {"name": line_name}
                }
                sbp_geojson["features"].append(feature)
            
            folium.GeoJson(
                sbp_geojson,
                name="SBP Lines",
                style_function=lambda x: {'color': '#00FFFF', 'weight': 2, 'opacity': 0.7},
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
                    popup=f"<b>{row['name']}</b>",
                    tooltip=row['name']
                ).add_to(m)
        except:
            pass
    
    # Add hazards
    if hazards is not None and len(hazards) > 0:
        try:
            gdf_haz = hazards.to_crs('EPSG:4326') if hazards.crs else hazards
            for idx, row in gdf_haz.iterrows():
                risk = row.get('risk', 'Unknown')
                hazard_type = row.get('hazard_type', 'Unknown')
                
                popup_html = f"""
                <div style="width:300px; font-family:Arial;">
                    <h3 style="color:{get_risk_color(risk)};">{hazard_type} - {row.get('id', '')}</h3>
                    <p><b>Name:</b> {row.get('name', 'N/A')}</p>
                    <p><b>Size:</b> {row.get('size', 'N/A')}</p>
                    <p><b>Detected By:</b> {row.get('detected_by', 'N/A')}</p>
                    <p><b>Distance:</b> {row.get('distance_to_turbine_m', 'N/A')}m</p>
                    <p><b>Risk:</b> <span style="color:{get_risk_color(risk)};">{risk}</span> ({row.get('risk_score', 'N/A')}/10)</p>
                    <p><b>Cost:</b> {row.get('cost', 'N/A')}</p>
                </div>
                """
                
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=folium.Popup(popup_html, max_width=320),
                    tooltip=f"{hazard_type}: {row.get('name', '')}",
                    icon=folium.Icon(
                        color='red' if risk == 'Critical' else 'orange' if risk == 'High' else 'beige',
                        icon='exclamation-triangle',
                        prefix='fa'
                    )
                ).add_to(m)
        except Exception as e:
            st.warning(f"Error adding hazards: {e}")
    
    folium.plugins.Fullscreen().add_to(m)
    folium.plugins.MousePosition().add_to(m)
    folium.LayerControl().add_to(m)
    
    return m

# ==============================================================================
# SIDEBAR
# ==============================================================================

st.sidebar.header("🎨 Settings")
basemap = st.sidebar.selectbox("Basemap", ['OpenStreetMap', 'Esri Satellite'], index=0)
mbes_colormap = st.sidebar.selectbox("MBES Colormap", ['ocean', 'viridis', 'seismic'], index=0)
quality = st.sidebar.radio("Quality", ["Fast (500px)", "Good (1000px)", "High (2000px)"], index=1)
max_pixels = {"Fast (500px)": 500, "Good (1000px)": 1000, "High (2000px)": 2000}[quality]

# PRE-CONFIGURED FILE IDs
FILE_IDS = {
    'sss': ['1-fd4WYSO3jAurneJNV_QzMJVx3F5rojM','1reqiNT6_XKdFc4LzjAM634CRe3qqReZP','1MmJAYU9O6bjqst0ufZW5s_7DZQirSr5Z','10XWv6wmnIX0zHDHTtIsOoM71JtByLNVb','10YZlXZ2JDp5f7ehg4xMdFkkZCD5NVHNI','12FLV4q_9X4EzGrqIWtGShCo-BUYrmRlG','134bxFTgfLwZYWvWhIa7swmhS2UGCV7M0','15X6Ho70GmLxlDHubSDnEfEQq85s0CKol','17dJXRk_VIZuQhULjj52-BqaUAnaaeeOH','17yLRua1a3AgBYuZC4_36D7x5-kfLON7L','1A2ZYFc6Mey_pJvBtB9gGXL5zxDeJtdRM','1AiH391YcyhizRgWSldH8Oijd6PzG9a9Y','1EC4BT0SBHsYf6iYXGYXUxhmKhYNbXoeY','1Ffz5h8qjND_jS-wA3QUxtQ0DUaoNj9wC','1GDU4aonNheXJ0pK7NsWqjLyXNDzE1NwM','1IJooNeDkLj4TqCxra7iOjFYtVU182e7I','1MLaLICacB1DpyPv1jkFq8SrY1tRKl7aN','1NYTPv-3PsWs7kjeer_uf4pfGMsbwTi0E','1N_Y_bmCTUuu15IYS9j-XLJNH28OyZD9H','1RZkpzxIQgrCYnWBGm_w42stbyzVmEPva','1VlAVkEbnTbFnto57sMbfFkbzU7p67LL7','1aVvPfIXoRDC2XqmDMG92fUIZtrFpIsJq','1bHSd6XzLDYIAwnQYlDPRo8FLnH8DDJnw','1cMBJlt0A6JwfMS7fhcvR7cnJ4468gLze','1cNNmCgY6iAbHMtGm2UPeJX_NeMnb2rQn','1cvLjbwFDPjD2avHzjMuwDGKYDjjXLT5v','1dhxT_QsbygFdLV9ZYUQ6wd5_2mZSb6e5','1eaS_7K8012AneqC5LkwmuLEqqJmPO1sq','1hg9wgSkhRIzYiIhCGq1xjbFMxzSz4pSm','1iDJWZcRz_zGbOTQpYN9U1V707X5xo3yv','1jNwjUx7zdHHFKxFtAXSbYLMrmVKDdRxS','1jqZLJ5xJhxdChh9SKlbahsLviqbqPzFx','1ldV6zBMMrWfovkNbV2bSSkHyZmPUKYlI','1nzPO4LXl6PJ5TffOe6c2pHJFmZzSUDfp','1sWFLzNsAo0ZQ_nbusrNm9I7DnfFh4TIq','1t0NXhHNdHQrwuMCzfiGu1CYb1z47-XVK','1w2ZwrKigqOHqXMRnyY4GD_jNn0VTCrWE','1wkcFrGXx8dVNf5gYMkEaNeIdvIRNazJz','1wmMgdqL-B56PI4sHQ-Fr4GFxp28ptb8U','1zso2rorqe_FXDXbMfHXl3vDRodD8H7fC'],
    'mbes': '1lE9X1S2Lqt3UxKgEJto5cURf1gTxOADr',
    'mag': '1jyYQ9ICEFjXxFAatFQvGb-9byu3ryq5P',
    'turbines': '18uYbX7OWZcqQfoBow6F_P4AmjptioeeO',
    'sbp': '1cZCoNX1t68X1BoiyikYKRAV0vzo_3pGO',
    'hazards': '1h3FUT5DYj3OAM3o3OtTUm-TmjCCoj8hM'
}

st.sidebar.markdown("---")
st.sidebar.header("📁 Quick Load")

# LOAD BUTTONS
if st.sidebar.button("🚀 Load SSS Tiles", use_container_width=True):
    prog = st.progress(0)
    stat = st.empty()
    count = 0
    for i, fid in enumerate(FILE_IDS['sss']):
        try:
            stat.text(f"Loading {i+1}/{len(FILE_IDS['sss'])}...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                download_from_gdrive(fid, tmp.name)
                img, bounds = tif_to_png_base64(tmp.name, 'gray', max_pixels, True, False)
                if img and bounds:
                    st.session_state.raster_layers.append((img, bounds))
                    count += 1
                os.unlink(tmp.name)
            prog.progress((i+1)/len(FILE_IDS['sss']))
        except:
            pass
    prog.empty()
    stat.empty()
    st.success(f"✅ Loaded {count} SSS tiles")
    st.rerun()

if st.sidebar.button("🗺️ Load MBES", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        try:
            download_from_gdrive(FILE_IDS['mbes'], tmp.name)
            img, bounds = tif_to_png_base64(tmp.name, mbes_colormap, max_pixels, False, False)
            if img and bounds:
                st.session_state.raster_layers.append((img, bounds))
                st.success("✅ MBES loaded")
                st.rerun()
            os.unlink(tmp.name)
        except Exception as e:
            st.error(f"Error: {e}")

if st.sidebar.button("🧲 Load Mag TIF", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        try:
            download_from_gdrive(FILE_IDS['mag'], tmp.name)
            img, bounds = tif_to_png_base64(tmp.name, 'seismic', max_pixels, False, True)
            if img and bounds:
                st.session_state.mag_tif_layer = (img, bounds)
                st.success("✅ Mag TIF loaded")
                st.rerun()
            os.unlink(tmp.name)
        except Exception as e:
            st.error(f"Error: {e}")

if st.sidebar.button("🔌 Load Turbines", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        try:
            download_from_gdrive(FILE_IDS['turbines'], tmp.name)
            st.session_state.turbines = gpd.read_file(tmp.name)
            st.success(f"✅ Loaded {len(st.session_state.turbines)} turbines")
            os.unlink(tmp.name)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if st.sidebar.button("🔊 Load SBP Lines", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        try:
            download_from_gdrive(FILE_IDS['sbp'], tmp.name)
            st.session_state.sbp_lines = gpd.read_file(tmp.name)
            st.success(f"✅ Loaded {len(st.session_state.sbp_lines)} lines")
            os.unlink(tmp.name)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if st.sidebar.button("⚠️ Load Hazards", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        try:
            download_from_gdrive(FILE_IDS['hazards'], tmp.name)
            st.session_state.hazards = gpd.read_file(tmp.name)
            st.success(f"✅ Loaded {len(st.session_state.hazards)} hazards")
            os.unlink(tmp.name)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if st.sidebar.button("🗑️ Clear All", use_container_width=True):
    st.session_state.raster_layers = []
    st.session_state.hazards = None
    st.session_state.turbines = None
    st.session_state.sbp_lines = None
    st.session_state.mag_tif_layer = None
    st.rerun()

# Layer toggles
st.sidebar.markdown("---")
st.sidebar.header("👁️ Layers")
show_sbp = st.sidebar.checkbox("Show SBP Lines", True)
show_mag = st.sidebar.checkbox("Show Mag TIF", True)

# Hazard filters
filtered_hazards = st.session_state.hazards
if filtered_hazards is not None and len(filtered_hazards) > 0:
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")
    
    types = sorted(filtered_hazards['hazard_type'].unique())
    sel_types = st.sidebar.multiselect("Hazard Types:", types, types)
    
    risks = ["Critical", "High", "Medium", "Low"]
    sel_risks = st.sidebar.multiselect("Risk Levels:", risks, risks)
    
    filtered_hazards = filtered_hazards[
        filtered_hazards['hazard_type'].isin(sel_types) &
        filtered_hazards['risk'].isin(sel_risks)
    ]

# ==============================================================================
# MAIN MAP
# ==============================================================================

has_data = (len(st.session_state.raster_layers) > 0 or
            (st.session_state.hazards is not None and len(st.session_state.hazards) > 0) or
            (st.session_state.turbines is not None and len(st.session_state.turbines) > 0))

if has_data:
    # Summary
    if filtered_hazards is not None and len(filtered_hazards) > 0:
        st.subheader("📊 Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Critical", len(filtered_hazards[filtered_hazards['risk']=='Critical']))
        c2.metric("High", len(filtered_hazards[filtered_hazards['risk']=='High']))
        c3.metric("Medium", len(filtered_hazards[filtered_hazards['risk']=='Medium']))
        c4.metric("Low", len(filtered_hazards[filtered_hazards['risk']=='Low']))
    
    # Map
    st.subheader("🗺️ Interactive Map")
    m = create_map(st.session_state.raster_layers, filtered_hazards,
                   st.session_state.turbines, st.session_state.sbp_lines,
                   st.session_state.mag_tif_layer, basemap, show_sbp, show_mag)
    st_folium(m, width=1400, height=700)
    
    # Hazard table
    if filtered_hazards is not None and len(filtered_hazards) > 0:
        st.subheader("📋 Hazard Register")
        cols = ['id', 'hazard_type', 'risk', 'distance_to_turbine_m', 'cost']
        st.dataframe(filtered_hazards[cols], use_container_width=True, height=300)
        
        csv = filtered_hazards.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"hazards_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
else:
    st.info("👆 Click buttons in sidebar to load data from Google Drive")
    st.markdown("""
    ## 📱 One-Click Demo
    
    All data is pre-configured from Google Drive:
    
    1. **🚀 Load SSS Tiles** - Sidescan sonar imagery (41 tiles)
    2. **🗺️ Load MBES** - Bathymetry
    3. **🧲 Load Mag TIF** - Magnetometer heatmap
    4. **🔌 Load Turbines** - Turbine locations
    5. **🔊 Load SBP Lines** - Survey coverage
    6. **⚠️ Load Hazards** - All detected hazards
    
    Perfect for mobile demos!
    """)
