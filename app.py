#!/usr/bin/env python3
"""
app.py - Marine Survey Viewer with Complete Risk Assessment
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
if 'mag_points' not in st.session_state:
    st.session_state.mag_points = []
if 'cable_route' not in st.session_state:
    st.session_state.cable_route = None

st.title("⚠️ Marine Survey Risk Assessment")

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
            
            bounds_wgs84 = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
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


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371000  # Earth radius in meters


def classify_risk(magnitude, distance_to_cable):
    """Classify risk based on magnitude and distance."""
    if magnitude > 50 and distance_to_cable < 50:
        return "Critical"
    elif magnitude > 50 or distance_to_cable < 50:
        return "High"
    elif magnitude > 20 or distance_to_cable < 100:
        return "Medium"
    else:
        return "Low"


def estimate_cost(risk, magnitude):
    """Estimate clearance/investigation cost."""
    if risk == "Critical":
        if magnitude > 100:
            return "£100,000 - £200,000"  # Likely UXO
        else:
            return "£50,000 - £75,000"  # Wreck clearance
    elif risk == "High":
        return "£20,000 - £40,000"  # ROV survey + possible clearance
    elif risk == "Medium":
        return "£5,000 - £15,000"  # ROV investigation
    else:
        return "£1,000 - £5,000"  # Monitoring


def recommend_action(risk, magnitude, distance):
    """Recommend action based on risk."""
    if risk == "Critical":
        if magnitude > 100:
            return "UXO clearance required"
        else:
            return "Immediate clearance or route alteration"
    elif risk == "High":
        return "ROV survey for classification, likely clearance"
    elif risk == "Medium":
        return "ROV investigation recommended"
    else:
        return "Monitor during installation"


def create_map(raster_layers, vector_layers, mag_points, cable_route, basemap_choice):
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
    
    for gdf, color, name in vector_layers:
        if gdf is not None:
            gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs and str(gdf.crs) != 'EPSG:4326' else gdf
            folium.GeoJson(gdf_wgs84, name=name,
                         style_function=lambda x, c=color: {'color': c, 'weight': 2}).add_to(m)
    
    # Draw cable route if provided
    if cable_route:
        folium.PolyLine(
            cable_route,
            color='blue',
            weight=3,
            opacity=0.8,
            popup='Cable Route'
        ).add_to(m)
    
    # Add mag points with risk classification
    if mag_points:
        for point in mag_points:
            magnitude = point.get('magnitude', 0)
            risk = point.get('risk', 'Low')
            distance = point.get('distance_to_cable', 999)
            
            # Color by risk
            if risk == "Critical":
                color = 'red'
            elif risk == "High":
                color = 'orange'
            elif risk == "Medium":
                color = 'yellow'
            else:
                color = 'green'
            
            popup_html = f"""<div style="width:250px;">
                <h4 style="color:{color};">Mag Anomaly</h4>
                <p><b>Magnitude:</b> {magnitude:.1f} nT</p>
                <p><b>Risk Level:</b> {risk}</p>
                <p><b>Distance to Cable:</b> {distance:.0f}m</p>
                <p><b>Action:</b> {point.get('action', 'Monitor')}</p>
                <p><b>Est. Cost:</b> {point.get('cost', 'TBD')}</p>
                </div>"""
            
            folium.CircleMarker(
                location=[point['lat'], point['lon']],
                radius=6 if risk in ["Critical", "High"] else 4,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{magnitude:.1f}nT - {risk}"
            ).add_to(m)
    
    folium.plugins.Fullscreen().add_to(m)
    folium.plugins.MousePosition().add_to(m)
    folium.plugins.MeasureControl(position='topleft').add_to(m)
    folium.LayerControl().add_to(m)
    
    return m


# DATA LOADING
st.sidebar.subheader("📡 Sidescan (SSS)")
sss_ids = st.sidebar.text_area("SSS File IDs", height=100)

if sss_ids and st.sidebar.button("🚀 Load SSS", type="primary"):
    sss_id_list = [fid.strip() for fid in sss_ids.strip().split('\n') if fid.strip()]
    
    progress_container = st.container()
    with progress_container:
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
                status_text.warning(f"Tile {i+1} failed: {e}")
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Loaded {success_count}/{len(sss_id_list)} SSS tiles!")

# MBES
st.sidebar.subheader("🗺️ MBES")
mbes_ids = st.sidebar.text_area("MBES File IDs", height=50)

if mbes_ids and st.sidebar.button("Load MBES"):
    for file_id in [fid.strip() for fid in mbes_ids.strip().split('\n') if fid.strip()]:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            try:
                download_from_gdrive(file_id, tmp.name)
                img, bounds = tif_to_png_base64(tmp.name, colormap=mbes_colormap, max_size=max_pixels)
                if img and bounds:
                    st.session_state.raster_layers.append((img, bounds))
                os.unlink(tmp.name)
            except Exception as e:
                st.error(f"Failed: {e}")
    st.success("✅ MBES loaded!")

# Mag TIF
st.sidebar.subheader("🧲 Mag TIF")
mag_tif_ids = st.sidebar.text_area("Mag TIF File IDs", height=50)

if mag_tif_ids and st.sidebar.button("Load Mag TIF"):
    for file_id in [fid.strip() for fid in mag_tif_ids.strip().split('\n') if fid.strip()]:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            try:
                download_from_gdrive(file_id, tmp.name)
                img, bounds = tif_to_png_base64(tmp.name, colormap='seismic',
                                               max_size=max_pixels, is_mag=True)
                if img and bounds:
                    st.session_state.raster_layers.append((img, bounds))
                os.unlink(tmp.name)
            except Exception as e:
                st.error(f"Failed: {e}")
    st.success("✅ Mag TIF loaded!")

# CABLE ROUTE
st.sidebar.subheader("🔌 Cable Route (GeoJSON)")
cable_id = st.sidebar.text_input("Cable Route File ID")

if cable_id and st.sidebar.button("Load Cable"):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
        try:
            download_from_gdrive(cable_id, tmp.name)
            gdf = gpd.read_file(tmp.name)
            gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs else gdf
            
            # Extract coordinates
            coords = []
            for geom in gdf_wgs84.geometry:
                if geom.geom_type == 'LineString':
                    coords = [(lat, lon) for lon, lat in geom.coords]
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        coords.extend([(lat, lon) for lon, lat in line.coords])
            
            st.session_state.cable_route = coords
            st.session_state.vector_layers.append((gdf_wgs84, 'blue', 'Cable'))
            st.success(f"✅ Cable route loaded ({len(coords)} points)")
            os.unlink(tmp.name)
        except Exception as e:
            st.error(f"Error: {e}")

# MAG CSV with Risk Classification
st.sidebar.subheader("🎯 Mag Targets CSV")

mag_csv_file = st.sidebar.file_uploader("Upload Mag Targets CSV", type=['csv'])

if mag_csv_file is not None:
    try:
        df = pd.read_csv(mag_csv_file)
        
        # Find columns
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            lat_col = 'Latitude'
            lon_col = 'Longitude'
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
            st.sidebar.success(f"✅ Found {len(df)} targets")
            
            # Threshold filter
            mag_threshold = st.sidebar.slider("Min Magnitude (nT)", 0,
                                             int(df[mag_col].max()), 5)
            df_filtered = df[df[mag_col] > mag_threshold]
            
            # Risk filter
            risk_filter = st.sidebar.multiselect(
                "Show Risk Levels",
                ["Critical", "High", "Medium", "Low"],
                default=["Critical", "High", "Medium", "Low"]
            )
            
            st.sidebar.info(f"Processing {len(df_filtered)} targets...")
            
            # Calculate distances and classify risks
            st.session_state.mag_points = []
            
            for _, row in df_filtered.iterrows():
                lat = row[lat_col]
                lon = row[lon_col]
                magnitude = row[mag_col]
                
                # Calculate distance to cable
                if st.session_state.cable_route:
                    min_distance = min(
                        calculate_distance(lat, lon, cable_lat, cable_lon)
                        for cable_lat, cable_lon in st.session_state.cable_route
                    )
                else:
                    min_distance = 999  # Unknown
                
                # Classify risk
                risk = classify_risk(magnitude, min_distance)
                
                # Filter by risk
                if risk not in risk_filter:
                    continue
                
                # Get action and cost
                action = recommend_action(risk, magnitude, min_distance)
                cost = estimate_cost(risk, magnitude)
                
                st.session_state.mag_points.append({
                    'lat': lat,
                    'lon': lon,
                    'magnitude': magnitude,
                    'distance_to_cable': min_distance,
                    'risk': risk,
                    'action': action,
                    'cost': cost
                })
            
            st.sidebar.success(f"✅ Showing {len(st.session_state.mag_points)} targets")
            
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if st.sidebar.button("🗑️ Clear All"):
    st.session_state.raster_layers = []
    st.session_state.vector_layers = []
    st.session_state.mag_points = []
    st.session_state.cable_route = None
    st.success("Cleared!")
    st.rerun()

# RISK SUMMARY
if st.session_state.mag_points:
    st.subheader("📊 Risk Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        critical = sum(1 for p in st.session_state.mag_points if p['risk'] == 'Critical')
        st.metric("🔴 Critical", critical)
    
    with col2:
        high = sum(1 for p in st.session_state.mag_points if p['risk'] == 'High')
        st.metric("🟠 High", high)
    
    with col3:
        medium = sum(1 for p in st.session_state.mag_points if p['risk'] == 'Medium')
        st.metric("🟡 Medium", medium)
    
    with col4:
        low = sum(1 for p in st.session_state.mag_points if p['risk'] == 'Low')
        st.metric("🟢 Low", low)
    
    with col5:
        # Calculate total cost (min values)
        total_cost = 0
        for p in st.session_state.mag_points:
            cost_str = p['cost'].replace('£', '').replace(',', '')
            if '-' in cost_str:
                min_cost = int(cost_str.split('-')[0].strip())
                total_cost += min_cost
        st.metric("💰 Min Cost", f"£{total_cost:,}")

# MAP
if st.session_state.raster_layers or st.session_state.mag_points:
    st.subheader("🗺️ Risk Assessment Map")
    
    m = create_map(st.session_state.raster_layers, st.session_state.vector_layers,
                  st.session_state.mag_points, st.session_state.cable_route, basemap)
    st_folium(m, width=1400, height=700, key="risk_map")

# HAZARD REGISTER
if st.session_state.mag_points:
    st.subheader("📋 Hazard Register")
    
    df_hazards = pd.DataFrame(st.session_state.mag_points)
    
    # Add ID column
    df_hazards.insert(0, 'ID', [f"MAG-{i+1:03d}" for i in range(len(df_hazards))])
    
    # Sort by risk
    risk_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    df_hazards['risk_order'] = df_hazards['risk'].map(risk_order)
    df_hazards = df_hazards.sort_values(['risk_order', 'magnitude'], ascending=[True, False])
    df_hazards = df_hazards.drop('risk_order', axis=1)
    
    # Display
    st.dataframe(
        df_hazards[['ID', 'magnitude', 'risk', 'distance_to_cable', 'action', 'cost']],
        use_container_width=True,
        height=400
    )
    
    # Export
    csv = df_hazards.to_csv(index=False)
    st.download_button(
        "📥 Download Hazard Register (CSV)",
        csv,
        f"hazard_register_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )
    
    # Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Actions Required")
        actions = df_hazards.groupby('action').size().reset_index(name='count')
        for _, row in actions.iterrows():
            st.write(f"• {row['action']}: **{row['count']}** items")
    
    with col2:
        st.markdown("### Cost Breakdown by Risk")
        for risk in ["Critical", "High", "Medium", "Low"]:
            risk_items = df_hazards[df_hazards['risk'] == risk]
            if len(risk_items) > 0:
                costs = []
                for cost_str in risk_items['cost']:
                    cost_clean = cost_str.replace('£', '').replace(',', '')
                    if '-' in cost_clean:
                        min_cost = int(cost_clean.split('-')[0].strip())
                        costs.append(min_cost)
                
                total = sum(costs)
                st.write(f"• {risk}: **£{total:,}** ({len(risk_items)} items)")

else:
    st.info("👆 Load survey data and mag targets to begin risk assessment")
