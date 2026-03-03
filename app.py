#!/usr/bin/env python3
"""
app.py - Marine Survey Viewer with Risk Features (Session State Fixed)
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

st.set_page_config(page_title="Marine Risk Viewer", layout="wide", page_icon="⚠️")

# Initialize session state FIRST
if 'raster_layers' not in st.session_state:
    st.session_state.raster_layers = []
if 'vector_layers' not in st.session_state:
    st.session_state.vector_layers = []
if 'show_demo' not in st.session_state:
    st.session_state.show_demo = True

st.title("⚠️ Marine Survey Risk Assessment Viewer")

# Sidebar
st.sidebar.header("📁 Data Mode")

data_mode = st.sidebar.radio(
    "Choose mode:",
    ["Demo (Risk Features)", "Load Your Data (MBES/SSS)"]
)

if data_mode == "Demo (Risk Features)":
    st.session_state.show_demo = True
else:
    st.session_state.show_demo = False

# ==============================================================================
# DEMO MODE - Risk Assessment
# ==============================================================================

if st.session_state.show_demo:
    st.sidebar.header("⚠️ Risk Filters")
    
    hazard_types = st.sidebar.multiselect(
        "Hazard Types",
        ["Wreck", "Boulder", "Mag Anomaly", "Buried Channel", "Mobile Sediment", "UXO"],
        default=["Wreck", "Boulder", "Mag Anomaly"]
    )
    
    risk_level = st.sidebar.multiselect(
        "Risk Level",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High"]
    )
    
    if "Mag Anomaly" in hazard_types:
        mag_threshold = st.sidebar.slider(
            "Mag Anomaly Threshold (nT)",
            min_value=0,
            max_value=500,
            value=50,
            step=10
        )
    else:
        mag_threshold = 50
    
    max_distance = st.sidebar.slider(
        "Max Distance to Cable (m)",
        min_value=0,
        max_value=500,
        value=100,
        step=10
    )
    
    show_infrastructure = st.sidebar.checkbox("🔌 Show Infrastructure", value=True)
    
    st.sidebar.header("🎨 Display")
    basemap = st.sidebar.selectbox(
        "Basemap",
        ['OpenStreetMap', 'Esri Satellite', 'CartoDB Positron'],
        index=0
    )
    
    # Demo data
    DEMO_HAZARDS = [
        {
            "id": "WRK-001", "type": "Wreck", "lat": 53.45, "lon": 2.15,
            "risk": "Critical", "confidence": "High", "distance_to_cable": 45,
            "description": "Partially buried wreck, 15m length",
            "action": "Clearance required",
            "estimated_cost": "£50,000 - £75,000",
            "detected_by": "SSS + Mag"
        },
        {
            "id": "BLD-002", "type": "Boulder", "lat": 53.46, "lon": 2.16,
            "risk": "High", "confidence": "High", "distance_to_cable": 23,
            "description": "Boulder field, 3-5m diameter",
            "action": "Route alteration recommended",
            "estimated_cost": "£10,000 - £20,000",
            "detected_by": "MBES + SSS"
        },
        {
            "id": "MAG-003", "type": "Mag Anomaly", "lat": 53.47, "lon": 2.17,
            "risk": "Medium", "confidence": "Medium", "distance_to_cable": 87,
            "magnitude": 125,
            "description": "Magnetic anomaly 125nT",
            "action": "Further investigation - ROV survey",
            "estimated_cost": "£5,000 - £15,000",
            "detected_by": "Magnetometer"
        },
        {
            "id": "BCH-004", "type": "Buried Channel", "lat": 53.44, "lon": 2.14,
            "risk": "High", "confidence": "High", "distance_to_cable": 12,
            "description": "Buried channel, 8m deep",
            "action": "Cable burial depth increase",
            "estimated_cost": "£30,000 - £50,000",
            "detected_by": "SBP", "sbp_line": "Line_045"
        },
        {
            "id": "UXO-006", "type": "UXO", "lat": 53.43, "lon": 2.13,
            "risk": "Critical", "confidence": "Low", "distance_to_cable": 67,
            "description": "Possible UXO",
            "action": "UXO specialist clearance",
            "estimated_cost": "£100,000 - £200,000",
            "detected_by": "Mag + SSS"
        }
    ]
    
    DEMO_CABLE = [[2.13, 53.43], [2.14, 53.44], [2.15, 53.45], 
                  [2.16, 53.46], [2.17, 53.47], [2.18, 53.48]]
    
    def get_risk_color(risk):
        colors = {"Critical": "#DC143C", "High": "#FF8C00", 
                 "Medium": "#FFD700", "Low": "#32CD32"}
        return colors.get(risk, "#808080")
    
    def get_hazard_icon(hazard_type):
        icons = {"Wreck": "ship", "Boulder": "circle", "Mag Anomaly": "magnet",
                "Buried Channel": "water", "UXO": "exclamation-triangle"}
        return icons.get(hazard_type, "question")
    
    def filter_hazards(hazards, types, risks, max_dist, mag_thresh):
        filtered = []
        for h in hazards:
            if h["type"] not in types: continue
            if h["risk"] not in risks: continue
            if h["distance_to_cable"] > max_dist: continue
            if h["type"] == "Mag Anomaly" and h.get("magnitude", 0) < mag_thresh: continue
            filtered.append(h)
        return filtered
    
    def create_risk_map(hazards, cable_route, show_infra, basemap_choice):
        if cable_route:
            lats = [c[1] for c in cable_route]
            lons = [c[0] for c in cable_route]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
        else:
            center_lat, center_lon = 53.45, 2.15
        
        if basemap_choice == 'Esri Satellite':
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)
            folium.TileLayer(tiles=tiles, attr='Esri').add_to(m)
        elif basemap_choice == 'CartoDB Positron':
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')
        else:
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')
        
        if show_infra and cable_route:
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in cable_route],
                color='blue', weight=3, opacity=0.8, popup='Export Cable Route'
            ).add_to(m)
            
            for lon, lat in cable_route:
                folium.Circle(
                    location=[lat, lon], radius=max_distance,
                    color='blue', fill=True, fillColor='blue',
                    fillOpacity=0.1, weight=1
                ).add_to(m)
        
        for hazard in hazards:
            popup_html = f"""
            <div style="width: 300px;">
                <h4 style="margin:0; color:{get_risk_color(hazard['risk'])};">
                    {hazard['type']} - {hazard['id']}
                </h4>
                <hr style="margin:5px 0;">
                <p style="margin:5px 0;"><b>Risk:</b> {hazard['risk']}</p>
                <p style="margin:5px 0;"><b>Distance:</b> {hazard['distance_to_cable']}m</p>
                <p style="margin:5px 0;"><b>Description:</b> {hazard['description']}</p>
                <p style="margin:5px 0;"><b>Action:</b> {hazard['action']}</p>
                <p style="margin:5px 0;"><b>Cost:</b> {hazard['estimated_cost']}</p>
            </div>
            """
            
            folium.Marker(
                location=[hazard['lat'], hazard['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(
                    color='red' if hazard['risk'] == 'Critical' else 
                          'orange' if hazard['risk'] == 'High' else 'beige',
                    icon=get_hazard_icon(hazard['type']), prefix='fa'
                ),
                tooltip=f"{hazard['type']}: {hazard['id']}"
            ).add_to(m)
        
        legend_html = f"""
        <div style="position: fixed; bottom: 50px; right: 50px; width: 200px; 
                    background-color: white; z-index:9999; border:2px solid grey; 
                    border-radius: 5px; padding: 10px">
            <h4 style="margin:0 0 10px 0;">Risk Legend</h4>
            <p style="margin:5px 0;"><span style="color:#DC143C;">⬤</span> Critical</p>
            <p style="margin:5px 0;"><span style="color:#FF8C00;">⬤</span> High</p>
            <p style="margin:5px 0;"><span style="color:#FFD700;">⬤</span> Medium</p>
            <p style="margin:5px 0;"><span style="color:#32CD32;">⬤</span> Low</p>
            <hr><p style="margin:5px 0;"><b>Total:</b> {len(hazards)}</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        plugins.MeasureControl(position='topleft', primary_length_unit='meters').add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MousePosition().add_to(m)
        
        return m
    
    # Filter and display
    filtered_hazards = filter_hazards(DEMO_HAZARDS, hazard_types, risk_level, 
                                     max_distance, mag_threshold)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 Critical", sum(1 for h in filtered_hazards if h['risk'] == 'Critical'))
    with col2:
        st.metric("🟠 High", sum(1 for h in filtered_hazards if h['risk'] == 'High'))
    with col3:
        st.metric("🟡 Medium", sum(1 for h in filtered_hazards if h['risk'] == 'Medium'))
    with col4:
        total = sum(int(h['estimated_cost'].split('£')[1].split(' -')[0].replace(',', '')) 
                   for h in filtered_hazards)
        st.metric("💰 Min Cost", f"£{total:,}")
    
    st.subheader("🗺️ Risk Assessment Map")
    
    if filtered_hazards or show_infrastructure:
        m = create_risk_map(filtered_hazards, 
                          DEMO_CABLE if show_infrastructure else None, 
                          show_infrastructure, basemap)
        st_folium(m, width=1400, height=600, key="risk_map")
    else:
        st.warning("⚠️ No hazards match filters")
    
    if filtered_hazards:
        st.subheader("📊 Hazard Register")
        df = pd.DataFrame(filtered_hazards)
        st.dataframe(
            df[['id', 'type', 'risk', 'distance_to_cable', 'action', 'estimated_cost']],
            use_container_width=True, height=300
        )
        
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"hazard_register_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

# ==============================================================================
# YOUR DATA MODE - MBES/SSS Loading
# ==============================================================================

else:
    st.sidebar.header("📁 Load Your Data")
    
    max_pixels = st.sidebar.select_slider(
        "Quality", options=[250, 500, 750, 1000], value=500
    )
    
    basemap = st.sidebar.selectbox(
        "Basemap", ['OpenStreetMap', 'Esri Satellite'], index=0
    )
    
    colormap = st.sidebar.selectbox(
        "Colormap", ['ocean', 'viridis', 'gray'], index=0
    )
    
    def download_from_gdrive(file_id, output_path):
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        with st.spinner("📥 Downloading..."):
            session = requests.Session()
            response = session.get(url, stream=True)
            
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    params = {'id': file_id, 'confirm': value}
                    response = session.get(url, params=params, stream=True)
            
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
        
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        st.success(f"✅ Downloaded ({file_size_mb:.1f} MB)")
        return output_path
    
    @st.cache_data(show_spinner=False)
    def tif_to_png_base64(file_path, colormap='viridis', max_size=500, is_sss=False):
        try:
            with rasterio.open(file_path) as src:
                orig_h, orig_w = src.height, src.width
                downsample = max(1, int(max(orig_h, orig_w) / max_size))
                out_h = orig_h // downsample
                out_w = orig_w // downsample
                
                if src.count >= 3 and is_sss:
                    data = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    for i in range(3):
                        data[:, :, i] = src.read(i + 1, out_shape=(out_h, out_w),
                                                resampling=rasterio.enums.Resampling.average)
                    is_rgb = True
                else:
                    data = src.read(1, out_shape=(out_h, out_w),
                                  resampling=rasterio.enums.Resampling.average).astype(np.float32)
                    is_rgb = False
                    nodata = src.nodata
                    if nodata is not None:
                        data[data == nodata] = np.nan
                
                bounds_wgs84 = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                
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
    
    def create_data_map(raster_layers, vector_layers, basemap_choice):
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
                    opacity=0.7, interactive=True, name=f"Raster {i+1}"
                ).add_to(m)
        
        for gdf, color, name in vector_layers:
            if gdf is not None:
                gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs and str(gdf.crs) != 'EPSG:4326' else gdf
                folium.GeoJson(gdf_wgs84, name=name,
                             style_function=lambda x, c=color: {'color': c, 'weight': 2}).add_to(m)
        
        folium.plugins.Fullscreen().add_to(m)
        folium.plugins.MousePosition().add_to(m)
        folium.LayerControl().add_to(m)
        
        return m
    
    # Data loading UI
    st.sidebar.subheader("Google Drive Files")
    
    mbes_id = st.sidebar.text_input("MBES File ID")
    if mbes_id and st.sidebar.button("Load MBES"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            try:
                download_from_gdrive(mbes_id, tmp.name)
                img, bounds = tif_to_png_base64(tmp.name, colormap=colormap, max_size=max_pixels)
                if img and bounds:
                    st.session_state.raster_layers.append((img, bounds))
                    st.success("✅ MBES added!")
                else:
                    st.error("❌ Failed to process MBES")
            finally:
                try:
                    os.unlink(tmp.name)
                except:
                    pass
    
    sss_ids = st.sidebar.text_area("SSS File IDs (one per line)", height=100)
    if sss_ids and st.sidebar.button("Load SSS"):
        for file_id in sss_ids.strip().split('\n'):
            file_id = file_id.strip()
            if file_id:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                    try:
                        download_from_gdrive(file_id, tmp.name)
                        img, bounds = tif_to_png_base64(tmp.name, colormap='gray', 
                                                       max_size=max_pixels, is_sss=True)
                        if img and bounds:
                            st.session_state.raster_layers.append((img, bounds))
                            st.success(f"✅ SSS tile added!")
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except:
                            pass
    
    vector_id = st.sidebar.text_input("Vector File ID (GeoJSON)")
    if vector_id and st.sidebar.button("Load Vector"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
            try:
                download_from_gdrive(vector_id, tmp.name)
                gdf = gpd.read_file(tmp.name)
                st.session_state.vector_layers.append((gdf, 'yellow', 'Tracklines'))
                st.success(f"✅ Vector added ({len(gdf)} features)")
            finally:
                try:
                    os.unlink(tmp.name)
                except:
                    pass
    
    if st.sidebar.button("🗑️ Clear All"):
        st.session_state.raster_layers = []
        st.session_state.vector_layers = []
        st.success("Cleared!")
        st.rerun()
    
    # Display map
    if st.session_state.raster_layers or st.session_state.vector_layers:
        st.subheader("🗺️ Survey Map")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Raster Layers", len(st.session_state.raster_layers))
        with col2:
            st.metric("Vector Layers", len(st.session_state.vector_layers))
        
        m = create_data_map(st.session_state.raster_layers, 
                          st.session_state.vector_layers, basemap)
        st_folium(m, width=1400, height=800, key="data_map")
    else:
        st.info("👆 Load data using sidebar")
