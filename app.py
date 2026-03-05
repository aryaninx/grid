#!/usr/bin/env python3
"""
app.py - Marine Survey Viewer with Interactive Risk Overlay
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
from shapely.geometry import Point, LineString
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

st.set_page_config(page_title="Marine Risk Viewer", layout="wide", page_icon="⚠️")

# Session state
if 'raster_layers' not in st.session_state:
    st.session_state.raster_layers = []
if 'vector_layers' not in st.session_state:
    st.session_state.vector_layers = []
if 'all_hazards' not in st.session_state:
    st.session_state.all_hazards = []
if 'cable_route' not in st.session_state:
    st.session_state.cable_route = None

st.title("⚠️ Marine Survey Risk Assessment")

# MODE SELECTION
st.sidebar.header("🎛️ View Mode")
view_mode = st.sidebar.radio(
    "Select View:",
    ["📊 Survey Data View", "⚠️ Risk Overlay View"],
    help="Survey view shows raw data. Risk overlay filters by risk level."
)

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

st.sidebar.header("🎨 Display")
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
    """Convert GeoTIFF to base64 PNG with better error handling."""
    try:
        # Try to open with rasterio
        with rasterio.open(file_path) as src:
            orig_h, orig_w = src.height, src.width
            
            if max_size >= 10000:
                downsample = 1
                out_h, out_w = orig_h, orig_w
            else:
                downsample = max(1, int(max(orig_h, orig_w) / max_size))
                out_h = orig_h // downsample
                out_w = orig_w // downsample
            
            # Read data
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
            
            # Transform bounds
            try:
                bounds_wgs84 = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            except Exception as e:
                st.warning(f"CRS transformation issue: {e}. Assuming WGS84.")
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
            
    except rasterio.errors.RasterioIOError as e:
        st.error(f"❌ Not a valid GeoTIFF: {e}")
        st.info("💡 Tip: Use GDAL to convert: gdal_translate -of GTiff input.tif output.tif")
        return None, None
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None


def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in meters."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371000


def classify_risk(magnitude, distance_to_cable, feature_type="mag"):
    """Classify risk level."""
    if feature_type == "mag":
        if magnitude > 50 and distance_to_cable < 50:
            return "Critical"
        elif magnitude > 50 or distance_to_cable < 50:
            return "High"
        elif magnitude > 20 or distance_to_cable < 100:
            return "Medium"
        else:
            return "Low"
    return "Low"


def estimate_cost(risk, magnitude, feature_type="mag"):
    """Estimate cost range."""
    if risk == "Critical":
        if magnitude > 100:
            return "£100,000 - £200,000"
        else:
            return "£50,000 - £75,000"
    elif risk == "High":
        return "£20,000 - £40,000"
    elif risk == "Medium":
        return "£5,000 - £15,000"
    else:
        return "£1,000 - £5,000"


def recommend_action(risk, magnitude, distance, feature_type="mag"):
    """Recommend action."""
    if risk == "Critical":
        if magnitude > 100:
            return "UXO clearance required"
        else:
            return "Immediate clearance or route alteration"
    elif risk == "High":
        return "ROV survey + likely clearance"
    elif risk == "Medium":
        return "ROV investigation"
    else:
        return "Monitor during installation"


def create_map(raster_layers, vector_layers, hazards, cable_route, basemap_choice, risk_filter=None):
    """Create map with optional risk filtering."""
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
    
    # Add rasters
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
    
    # Add vectors
    for gdf, color, name in vector_layers:
        if gdf is not None:
            gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs and str(gdf.crs) != 'EPSG:4326' else gdf
            folium.GeoJson(gdf_wgs84, name=name,
                         style_function=lambda x, c=color: {'color': c, 'weight': 2}).add_to(m)
    
    # Add cable route
    if cable_route:
        folium.PolyLine(cable_route, color='blue', weight=3, opacity=0.8, popup='Cable Route').add_to(m)
    
    # Filter hazards by risk level
    if risk_filter:
        filtered_hazards = [h for h in hazards if h['risk'] in risk_filter]
    else:
        filtered_hazards = hazards
    
    # Add hazard markers with detailed popups
    for hazard in filtered_hazards:
        risk = hazard['risk']
        magnitude = hazard.get('magnitude', 0)
        distance = hazard.get('distance_to_cable', 999)
        
        # Color by risk
        if risk == "Critical":
            color = 'red'
        elif risk == "High":
            color = 'orange'
        elif risk == "Medium":
            color = 'yellow'
        else:
            color = 'green'
        
        # Detailed popup with ALL information
        popup_html = f"""
        <div style="width:300px; font-family: Arial;">
            <h3 style="margin:0; color:{color}; border-bottom:2px solid {color}; padding-bottom:5px;">
                {hazard.get('type', 'Mag Anomaly')} - {hazard.get('id', 'N/A')}
            </h3>
            
            <div style="margin-top:10px;">
                <h4 style="margin:5px 0; color:#333;">📊 Classification</h4>
                <p style="margin:5px 0;"><b>Risk Level:</b> <span style="color:{color}; font-weight:bold;">{risk}</span></p>
                <p style="margin:5px 0;"><b>Magnitude:</b> {magnitude:.1f} nT</p>
                <p style="margin:5px 0;"><b>Distance to Cable:</b> {distance:.0f}m</p>
            </div>
            
            <div style="margin-top:10px; background:#f0f0f0; padding:8px; border-radius:4px;">
                <h4 style="margin:5px 0; color:#333;">🎯 Recommended Action</h4>
                <p style="margin:5px 0;">{hazard.get('action', 'Monitor')}</p>
            </div>
            
            <div style="margin-top:10px; background:#fff3cd; padding:8px; border-radius:4px;">
                <h4 style="margin:5px 0; color:#333;">💰 Estimated Cost</h4>
                <p style="margin:5px 0; font-weight:bold;">{hazard.get('cost', 'TBD')}</p>
            </div>
            
            <div style="margin-top:10px; font-size:11px; color:#666;">
                <p style="margin:2px 0;">Lat: {hazard['lat']:.6f}</p>
                <p style="margin:2px 0;">Lon: {hazard['lon']:.6f}</p>
            </div>
        </div>
        """
        
        # Tooltip on hover - summary info
        tooltip_text = f"""
        <div style="font-family:Arial; padding:5px;">
            <b>{hazard.get('type', 'Mag Anomaly')}</b><br>
            Risk: <b style="color:{color}">{risk}</b><br>
            {magnitude:.1f} nT | {distance:.0f}m from cable<br>
            <i>{hazard.get('action', 'Monitor')}</i><br>
            <b>{hazard.get('cost', 'TBD')}</b>
        </div>
        """
        
        folium.CircleMarker(
            location=[hazard['lat'], hazard['lon']],
            radius=8 if risk in ["Critical", "High"] else 5,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=folium.Tooltip(tooltip_text)
        ).add_to(m)
    
    # Add legend
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; right: 50px; width: 200px; 
                background-color: white; z-index:9999; border:2px solid grey; 
                border-radius: 5px; padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin:0 0 10px 0;">Risk Legend</h4>
        <p style="margin:5px 0;"><span style="color:#DC143C; font-size:20px;">●</span> Critical</p>
        <p style="margin:5px 0;"><span style="color:#FF8C00; font-size:20px;">●</span> High</p>
        <p style="margin:5px 0;"><span style="color:#FFD700; font-size:20px;">●</span> Medium</p>
        <p style="margin:5px 0;"><span style="color:#32CD32; font-size:20px;">●</span> Low</p>
        <hr style="margin:10px 0;">
        <p style="margin:5px 0; font-size:12px;"><b>Showing:</b> {len(filtered_hazards)} hazards</p>
        <p style="margin:5px 0; font-size:11px; color:#666;"><i>Hover for details<br>Click for full info</i></p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    folium.plugins.Fullscreen().add_to(m)
    folium.plugins.MousePosition().add_to(m)
    folium.plugins.MeasureControl(position='topleft').add_to(m)
    folium.LayerControl().add_to(m)
    
    return m


# DATA LOADING
st.sidebar.subheader("📡 Sidescan (SSS)")
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
            except Exception as e:
                status_text.warning(f"Tile {i+1} failed")
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ {success_count}/{len(sss_id_list)} SSS tiles loaded!")

# MBES
st.sidebar.subheader("🗺️ MBES")
mbes_ids = st.sidebar.text_area("MBES File IDs", height=40)

if mbes_ids and st.sidebar.button("Load MBES"):
    for file_id in [fid.strip() for fid in mbes_ids.strip().split('\n') if fid.strip()]:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            try:
                download_from_gdrive(file_id, tmp.name)
                img, bounds = tif_to_png_base64(tmp.name, colormap=mbes_colormap, max_size=max_pixels)
                if img and bounds:
                    st.session_state.raster_layers.append((img, bounds))
                    st.success("✅ MBES loaded!")
                os.unlink(tmp.name)
            except Exception as e:
                st.error(f"Failed to load MBES")

# Cable Route
st.sidebar.subheader("🔌 Cable Route")
cable_id = st.sidebar.text_input("Cable GeoJSON File ID")

if cable_id and st.sidebar.button("Load Cable"):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
        try:
            download_from_gdrive(cable_id, tmp.name)
            gdf = gpd.read_file(tmp.name)
            gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs else gdf
            
            coords = []
            for geom in gdf_wgs84.geometry:
                if geom.geom_type == 'LineString':
                    coords = [(lat, lon) for lon, lat in geom.coords]
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        coords.extend([(lat, lon) for lon, lat in line.coords])
            
            st.session_state.cable_route = coords
            st.session_state.vector_layers.append((gdf_wgs84, 'blue', 'Cable'))
            st.success(f"✅ Cable route loaded!")
            os.unlink(tmp.name)
        except Exception as e:
            st.error(f"Error: {e}")

# Mag CSV - Process into hazards
st.sidebar.subheader("🎯 Mag Targets CSV")
mag_csv_file = st.sidebar.file_uploader("Upload Mag CSV", type=['csv'])

if mag_csv_file is not None:
    try:
        df = pd.read_csv(mag_csv_file)
        
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            lat_col, lon_col = 'Latitude', 'Longitude'
        else:
            lat_col = lon_col = None
            for col in df.columns:
                col_lower = col.lower()
                if 'lat' in col_lower and lat_col is None:
                    lat_col = col
                if 'lon' in col_lower and lon_col is None:
                    lon_col = col
        
        mag_col = None
        for col in df.columns:
            if 'nt' in col.lower() or 'mag' in col.lower():
                mag_col = col
                break
        
        if lat_col and lon_col and mag_col:
            mag_threshold = st.sidebar.slider("Min Magnitude (nT)", 0, int(df[mag_col].max()), 5)
            df_filtered = df[df[mag_col] > mag_threshold]
            
            st.sidebar.info(f"Processing {len(df_filtered)} targets...")
            
            # Process all hazards
            st.session_state.all_hazards = []
            
            for idx, row in df_filtered.iterrows():
                lat = row[lat_col]
                lon = row[lon_col]
                magnitude = row[mag_col]
                
                if st.session_state.cable_route:
                    min_distance = min(
                        calculate_distance(lat, lon, cable_lat, cable_lon)
                        for cable_lat, cable_lon in st.session_state.cable_route
                    )
                else:
                    min_distance = 999
                
                risk = classify_risk(magnitude, min_distance)
                action = recommend_action(risk, magnitude, min_distance)
                cost = estimate_cost(risk, magnitude)
                
                st.session_state.all_hazards.append({
                    'id': f"MAG-{idx+1:03d}",
                    'type': 'Mag Anomaly',
                    'lat': lat,
                    'lon': lon,
                    'magnitude': magnitude,
                    'distance_to_cable': min_distance,
                    'risk': risk,
                    'action': action,
                    'cost': cost
                })
            
            st.sidebar.success(f"✅ {len(st.session_state.all_hazards)} hazards processed!")
            
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if st.sidebar.button("🗑️ Clear All"):
    st.session_state.raster_layers = []
    st.session_state.vector_layers = []
    st.session_state.all_hazards = []
    st.session_state.cable_route = None
    st.success("Cleared!")
    st.rerun()

# RISK OVERLAY MODE
if view_mode == "⚠️ Risk Overlay View":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ Risk Filter")
    
    risk_filter = st.sidebar.multiselect(
        "Show Risk Levels:",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High"],
        help="Select which risk levels to display on the map"
    )
    
    # Summary
    if st.session_state.all_hazards:
        filtered = [h for h in st.session_state.all_hazards if h['risk'] in risk_filter]
        
        st.subheader(f"📊 Risk Summary - Showing {risk_filter}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            critical = sum(1 for h in filtered if h['risk'] == 'Critical')
            st.metric("🔴 Critical", critical)
        
        with col2:
            high = sum(1 for h in filtered if h['risk'] == 'High')
            st.metric("🟠 High", high)
        
        with col3:
            medium = sum(1 for h in filtered if h['risk'] == 'Medium')
            st.metric("🟡 Medium", medium)
        
        with col4:
            low = sum(1 for h in filtered if h['risk'] == 'Low')
            st.metric("🟢 Low", low)
        
        with col5:
            total_cost = 0
            for h in filtered:
                cost_str = h['cost'].replace('£', '').replace(',', '')
                if '-' in cost_str:
                    min_cost = int(cost_str.split('-')[0].strip())
                    total_cost += min_cost
            st.metric("💰 Min Cost", f"£{total_cost:,}")
        
        st.subheader("🗺️ Risk Overlay Map")
        st.info("💡 **Hover** over markers for quick info | **Click** for detailed breakdown")
        
        m = create_map(st.session_state.raster_layers, st.session_state.vector_layers,
                      st.session_state.all_hazards, st.session_state.cable_route, basemap, risk_filter)
        st_folium(m, width=1400, height=700)
        
        # Summary tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Actions Required")
            df_filtered = pd.DataFrame(filtered)
            actions = df_filtered.groupby('action').size().reset_index(name='count')
            for _, row in actions.iterrows():
                st.write(f"• **{row['action']}**: {row['count']} items")
        
        with col2:
            st.markdown("### 💰 Cost Breakdown")
            for risk in ["Critical", "High", "Medium", "Low"]:
                risk_items = [h for h in filtered if h['risk'] == risk]
                if risk_items:
                    costs = []
                    for h in risk_items:
                        cost_str = h['cost'].replace('£', '').replace(',', '')
                        if '-' in cost_str:
                            costs.append(int(cost_str.split('-')[0].strip()))
                    st.write(f"• **{risk}**: £{sum(costs):,} ({len(risk_items)} items)")
    else:
        st.info("Load mag targets CSV to enable risk overlay mode")

# SURVEY DATA MODE
else:
    if st.session_state.raster_layers or st.session_state.all_hazards:
        st.subheader("🗺️ Survey Data Map")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rasters", len(st.session_state.raster_layers))
        with col2:
            st.metric("Hazards", len(st.session_state.all_hazards))
        
        m = create_map(st.session_state.raster_layers, st.session_state.vector_layers,
                      st.session_state.all_hazards, st.session_state.cable_route, basemap)
        st_folium(m, width=1400, height=700)
    else:
        st.info("👆 Load survey data to begin")
