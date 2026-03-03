#!/usr/bin/env python3
"""
app.py - Marine Survey Viewer with Risk Features
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

st.title("⚠️ Marine Survey Risk Assessment Viewer")

# Sidebar
st.sidebar.header("📁 Data Layers")

# Layer toggles
show_bathymetry = st.sidebar.checkbox("🗺️ Bathymetry (MBES)", value=True)
show_sidescan = st.sidebar.checkbox("📡 Sidescan (SSS)", value=True)
show_infrastructure = st.sidebar.checkbox("🔌 Infrastructure", value=True)
show_hazards = st.sidebar.checkbox("⚠️ Hazards", value=True)

st.sidebar.header("⚠️ Risk Filters")

# Hazard type filters
hazard_types = st.sidebar.multiselect(
    "Hazard Types",
    ["Wreck", "Boulder", "Mag Anomaly", "Buried Channel", "Mobile Sediment", "UXO"],
    default=["Wreck", "Boulder", "Mag Anomaly"]
)

# Risk level filter
risk_level = st.sidebar.multiselect(
    "Risk Level",
    ["Critical", "High", "Medium", "Low"],
    default=["Critical", "High"]
)

# Mag anomaly threshold
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

# Distance to infrastructure
max_distance = st.sidebar.slider(
    "Max Distance to Cable (m)",
    min_value=0,
    max_value=500,
    value=100,
    step=10,
    help="Only show hazards within this distance"
)

st.sidebar.header("🎨 Display")
basemap = st.sidebar.selectbox(
    "Basemap",
    ['OpenStreetMap', 'Esri Satellite', 'CartoDB Positron'],
    index=0
)


# Sample hazard data structure
DEMO_HAZARDS = [
    {
        "id": "WRK-001",
        "type": "Wreck",
        "lat": 53.45,
        "lon": 2.15,
        "risk": "Critical",
        "confidence": "High",
        "distance_to_cable": 45,
        "description": "Partially buried wreck, 15m length",
        "action": "Clearance required",
        "estimated_cost": "£50,000 - £75,000",
        "detected_by": "SSS + Mag"
    },
    {
        "id": "BLD-002", 
        "type": "Boulder",
        "lat": 53.46,
        "lon": 2.16,
        "risk": "High",
        "confidence": "High",
        "distance_to_cable": 23,
        "description": "Boulder field, 3-5m diameter",
        "action": "Route alteration recommended",
        "estimated_cost": "£10,000 - £20,000",
        "detected_by": "MBES + SSS"
    },
    {
        "id": "MAG-003",
        "type": "Mag Anomaly",
        "lat": 53.47,
        "lon": 2.17,
        "risk": "Medium",
        "confidence": "Medium",
        "distance_to_cable": 87,
        "magnitude": 125,
        "description": "Magnetic anomaly 125nT",
        "action": "Further investigation - ROV survey",
        "estimated_cost": "£5,000 - £15,000",
        "detected_by": "Magnetometer"
    },
    {
        "id": "BCH-004",
        "type": "Buried Channel",
        "lat": 53.44,
        "lon": 2.14,
        "risk": "High",
        "confidence": "High",
        "distance_to_cable": 12,
        "description": "Buried channel, 8m deep, mobile sediment",
        "action": "Cable burial depth increase",
        "estimated_cost": "£30,000 - £50,000",
        "detected_by": "SBP",
        "sbp_line": "Line_045"
    },
    {
        "id": "MOB-005",
        "type": "Mobile Sediment",
        "lat": 53.48,
        "lon": 2.18,
        "risk": "Medium",
        "confidence": "Medium",
        "distance_to_cable": 156,
        "description": "Area of mobile sand waves, 2-4m height",
        "action": "Monitor during installation",
        "estimated_cost": "£2,000 - £5,000",
        "detected_by": "MBES time-series"
    },
    {
        "id": "UXO-006",
        "type": "UXO",
        "lat": 53.43,
        "lon": 2.13,
        "risk": "Critical",
        "confidence": "Low",
        "distance_to_cable": 67,
        "description": "Possible UXO - further investigation required",
        "action": "UXO specialist clearance",
        "estimated_cost": "£100,000 - £200,000",
        "detected_by": "Mag + SSS"
    }
]

# Demo cable route
DEMO_CABLE = [
    [2.13, 53.43],
    [2.14, 53.44],
    [2.15, 53.45],
    [2.16, 53.46],
    [2.17, 53.47],
    [2.18, 53.48]
]


def get_risk_color(risk):
    """Get color for risk level."""
    colors = {
        "Critical": "#DC143C",
        "High": "#FF8C00",
        "Medium": "#FFD700",
        "Low": "#32CD32"
    }
    return colors.get(risk, "#808080")


def get_hazard_icon(hazard_type):
    """Get icon for hazard type."""
    icons = {
        "Wreck": "ship",
        "Boulder": "circle",
        "Mag Anomaly": "magnet",
        "Buried Channel": "water",
        "Mobile Sediment": "water",
        "UXO": "exclamation-triangle"
    }
    return icons.get(hazard_type, "question")


def calculate_distance(lat1, lon1, lat2, lon2):
    """Simple distance calculation (rough approximation in meters)."""
    from math import radians, cos, sin, asin, sqrt
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000
    return c * r


def filter_hazards(hazards, types, risks, max_dist, mag_thresh):
    """Filter hazards based on criteria."""
    filtered = []
    
    for h in hazards:
        if h["type"] not in types:
            continue
        if h["risk"] not in risks:
            continue
        if h["distance_to_cable"] > max_dist:
            continue
        if h["type"] == "Mag Anomaly" and h.get("magnitude", 0) < mag_thresh:
            continue
        
        filtered.append(h)
    
    return filtered


def create_risk_map(hazards, cable_route, show_infra):
    """Create Folium map with risk visualization."""
    
    if cable_route:
        lats = [coord[1] for coord in cable_route]
        lons = [coord[0] for coord in cable_route]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    else:
        center_lat, center_lon = 53.45, 2.15
    
    if basemap == 'Esri Satellite':
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)
        folium.TileLayer(tiles=tiles, attr='Esri').add_to(m)
    elif basemap == 'CartoDB Positron':
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')
    
    if show_infra and cable_route:
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in cable_route],
            color='blue',
            weight=3,
            opacity=0.8,
            popup='Export Cable Route'
        ).add_to(m)
        
        for lon, lat in cable_route:
            folium.Circle(
                location=[lat, lon],
                radius=max_distance,
                color='blue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.1,
                weight=1,
                popup=f'Cable buffer zone ({max_distance}m)'
            ).add_to(m)
    
    for hazard in hazards:
        popup_html = f"""
        <div style="width: 300px;">
            <h4 style="margin:0; color:{get_risk_color(hazard['risk'])};">
                {hazard['type']} - {hazard['id']}
            </h4>
            <hr style="margin:5px 0;">
            <p style="margin:5px 0;"><b>Risk Level:</b> {hazard['risk']}</p>
            <p style="margin:5px 0;"><b>Confidence:</b> {hazard['confidence']}</p>
            <p style="margin:5px 0;"><b>Distance to Cable:</b> {hazard['distance_to_cable']}m</p>
            <p style="margin:5px 0;"><b>Description:</b> {hazard['description']}</p>
            <hr style="margin:5px 0;">
            <p style="margin:5px 0;"><b>Recommended Action:</b><br>{hazard['action']}</p>
            <p style="margin:5px 0;"><b>Estimated Cost:</b> {hazard['estimated_cost']}</p>
            <p style="margin:5px 0;"><b>Detected By:</b> {hazard['detected_by']}</p>
            {f'<p style="margin:5px 0;"><b>SBP Line:</b> {hazard["sbp_line"]}</p>' if 'sbp_line' in hazard else ''}
            {f'<p style="margin:5px 0;"><b>Magnitude:</b> {hazard["magnitude"]}nT</p>' if 'magnitude' in hazard else ''}
        </div>
        """
        
        folium.Marker(
            location=[hazard['lat'], hazard['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(
                color='red' if hazard['risk'] == 'Critical' else 
                      'orange' if hazard['risk'] == 'High' else
                      'beige' if hazard['risk'] == 'Medium' else 'green',
                icon=get_hazard_icon(hazard['type']),
                prefix='fa'
            ),
            tooltip=f"{hazard['type']}: {hazard['id']}"
        ).add_to(m)
        
        if show_infra and cable_route:
            min_dist = float('inf')
            nearest_point = None
            for lon, lat in cable_route:
                dist = calculate_distance(hazard['lat'], hazard['lon'], lat, lon)
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = (lat, lon)
            
            if nearest_point:
                folium.PolyLine(
                    locations=[[hazard['lat'], hazard['lon']], nearest_point],
                    color=get_risk_color(hazard['risk']),
                    weight=2,
                    opacity=0.5,
                    dashArray='5, 5'
                ).add_to(m)
    
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; 
                background-color: white; z-index:9999; 
                border:2px solid grey; border-radius: 5px;
                padding: 10px">
        <h4 style="margin:0 0 10px 0;">Risk Legend</h4>
        <p style="margin:5px 0;"><span style="color:#DC143C;">⬤</span> Critical</p>
        <p style="margin:5px 0;"><span style="color:#FF8C00;">⬤</span> High</p>
        <p style="margin:5px 0;"><span style="color:#FFD700;">⬤</span> Medium</p>
        <p style="margin:5px 0;"><span style="color:#32CD32;">⬤</span> Low</p>
        <hr>
        <p style="margin:5px 0; font-size:12px;"><b>Total Hazards:</b> {len(hazards)}</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    plugins.MeasureControl(position='topleft', primary_length_unit='meters').add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MousePosition().add_to(m)
    
    return m


# Filter hazards
filtered_hazards = filter_hazards(
    DEMO_HAZARDS,
    hazard_types,
    risk_level,
    max_distance,
    mag_threshold
)

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    critical = sum(1 for h in filtered_hazards if h['risk'] == 'Critical')
    st.metric("🔴 Critical", critical)

with col2:
    high = sum(1 for h in filtered_hazards if h['risk'] == 'High')
    st.metric("🟠 High", high)

with col3:
    medium = sum(1 for h in filtered_hazards if h['risk'] == 'Medium')
    st.metric("🟡 Medium", medium)

with col4:
    total_cost_min = sum(
        int(h['estimated_cost'].split('£')[1].split(' -')[0].replace(',', ''))
        for h in filtered_hazards
    )
    st.metric("💰 Min Cost", f"£{total_cost_min:,}")

# Create map
st.subheader("🗺️ Risk Assessment Map")

if filtered_hazards or show_infrastructure:
    m = create_risk_map(filtered_hazards, DEMO_CABLE if show_infrastructure else None, show_infrastructure)
    st_folium(m, width=1400, height=600, key="risk_map")
else:
    st.warning("⚠️ No hazards match current filters")

# Hazard table
st.subheader("📊 Hazard Register")

if filtered_hazards:
    df = pd.DataFrame(filtered_hazards)
    
    sort_by = st.selectbox(
        "Sort by:",
        ["Risk Level", "Distance to Cable", "Estimated Cost", "Type"]
    )
    
    if sort_by == "Risk Level":
        risk_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        df['risk_order'] = df['risk'].map(risk_order)
        df = df.sort_values('risk_order').drop('risk_order', axis=1)
    elif sort_by == "Distance to Cable":
        df = df.sort_values('distance_to_cable')
    elif sort_by == "Estimated Cost":
        df['cost_min'] = df['estimated_cost'].apply(
            lambda x: int(x.split('£')[1].split(' -')[0].replace(',', ''))
        )
        df = df.sort_values('cost_min', ascending=False).drop('cost_min', axis=1)
    else:
        df = df.sort_values('type')
    
    st.dataframe(
        df[['id', 'type', 'risk', 'confidence', 'distance_to_cable', 
            'action', 'estimated_cost', 'detected_by']],
        use_container_width=True,
        height=400
    )
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Hazard Register (CSV)",
        data=csv,
        file_name=f"hazard_register_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    st.subheader("📈 Risk Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Actions Required")
        actions = df.groupby('action').size().reset_index(name='count')
        for _, row in actions.iterrows():
            st.write(f"• {row['action']}: **{row['count']}** items")
    
    with col2:
        st.markdown("### Detection Methods")
        methods = df.groupby('detected_by').size().reset_index(name='count')
        for _, row in methods.iterrows():
            st.write(f"• {row['detected_by']}: **{row['count']}** items")

else:
    st.info("No hazards to display")

with st.expander("ℹ️ How to Use This Tool"):
    st.markdown("""
    ## Risk Assessment Workflow
    
    ### 1. Filter Hazards
    - Use sidebar to select hazard types
    - Set risk level threshold  
    - Adjust distance to infrastructure
    - For mag anomalies: set nT threshold
    
    ### 2. Review Map
    - Click markers for detailed information
    - Dashed lines show distance to cable
    - Blue circles show buffer zones
    - Measure distances with ruler tool
    
    ### 3. Analyze Table
    - Sort by risk, distance, or cost
    - Review recommended actions
    - Check detection confidence
    - Export to CSV for reports
    
    ### 4. Plan Mitigation
    - Critical items: immediate action
    - High items: clearance/route change
    - Medium items: monitor or investigate
    - Low items: document for reference
    """)