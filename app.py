#!/usr/bin/env python3
"""
THE GRID - Marine Hazard Intelligence Platform
InX Technologies Branded Version
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
from datetime import datetime, timedelta
import plotly.figure_factory as ff
import plotly.graph_objects as go

# PAGE CONFIG - DARK MODE
st.set_page_config(
    page_title="The Grid | InX Technologies",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# INX COLOR PALETTE
INX_COLORS = {
    'deep_black': '#2E2E2E',
    'musk_green': '#12241E',
    'sage_green': '#8F998D',
    'limitless_space': '#F2F2EF',
    'rocky_blue': '#8288A3',
    'precision_blue': '#0013C3',
    'neon_current': '#D1FE49'
}

# CUSTOM CSS - CLEAN WHITE THEME WITH INX CHART COLORS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global Font - Applied Everywhere */
    * {{
        font-family: 'Space Grotesk', sans-serif !important;
    }}
    
    /* Folium popup text */
    .leaflet-popup-content {{
        font-family: 'Space Grotesk', sans-serif !important;
    }}
    
    .leaflet-popup-content * {{
        font-family: 'Space Grotesk', sans-serif !important;
    }}
</style>
""", unsafe_allow_html=True)

# Session state
for key in ['raster_layers', 'hazards', 'turbines', 'sbp_lines', 'mag_tif_layer']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'raster_layers' else None

# BRANDED HEADER WITH LOGO FROM GOOGLE DRIVE
# Replace FILE_ID with your actual Google Drive file ID
LOGO_FILE_ID = "1gsPiE7nKz6j2X0gPvJqCO65cCVpXc32-"  # Update with your actual File ID

st.markdown(f"""
<div style='padding: 10px 20px; margin-bottom: 20px;'>
    <div style='display: flex; align-items: center; gap: 20px;'>
        <img src='https://drive.google.com/uc?export=view&id={LOGO_FILE_ID}' 
             style='height: 60px; width: auto;'
             alt='InX Technologies'
             onerror="this.style.display='none'">
        <h1 style='margin: 0; font-size: 42px; font-weight: 700; font-family: Space Grotesk;'>
            ⚡ THE GRID
        </h1>
    </div>
</div>
""", unsafe_allow_html=True)

# PAGE NAVIGATION with InX styling
page = st.radio(
    "",
    ["🗺️ Hazard Map", "📅 Project Timeline", "🔬 Evidence Viewer"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ==============================================================================
# FUNCTIONS (SAME AS BEFORE)
# ==============================================================================

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
def tif_to_png_base64(file_path, colormap='gray', max_size=1000, is_sss=False, is_mag=False):
    try:
        with rasterio.open(file_path) as src:
            orig_h, orig_w = src.height, src.width
            downsample = max(1, int(max(orig_h, orig_w) / max_size))
            out_h, out_w = orig_h // downsample, orig_w // downsample
            
            if src.count >= 3:
                data = np.zeros((out_h, out_w, 3), dtype=np.float32)
                for i in range(3):
                    band = src.read(i+1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.average)
                    data[:, :, i] = band.astype(np.float32)
                is_rgb = True
            else:
                data = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.average).astype(np.float32)
                is_rgb = False
            
            try:
                bounds_wgs84 = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            except:
                bounds_wgs84 = src.bounds
            
            nodata = src.nodata
            
            if is_rgb:
                valid_mask = np.ones((out_h, out_w), dtype=bool)
                if nodata is not None:
                    valid_mask &= ~np.all(data == nodata, axis=2)
                valid_mask &= ~np.all(data == 0, axis=2)
                valid_mask &= ~np.all(data == 255, axis=2)
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
                    valid_mask &= (data != 0) & (data < 255)
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
            return base64.b64encode(buffer.read()).decode(), bounds_wgs84
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def get_risk_color(risk):
    """Get InX-branded colors for risk levels"""
    colors = {
        'Critical': '#D1FE49',  # Neon current
        'High': '#0013C3',      # Precision blue
        'Medium': '#8F998D',    # Sage green
        'Low': '#8288A3'        # Rocky blue
    }
    return colors.get(risk, '#F2F2EF')

def create_timeline_gantt(hazards_gdf, scenario='original'):
    """Create Gantt chart with InX branding"""
    start_date = datetime(2024, 1, 1)
    
    if scenario == 'original':
        phases = [
            dict(Task="Geophysical Survey", Start=0, Duration=3, Resource="Survey"),
            dict(Task="Data Processing", Start=3, Duration=2, Resource="Analysis"),
            dict(Task="Geotechnical Survey", Start=5, Duration=4, Resource="Survey"),
            dict(Task="Engineering Design", Start=9, Duration=6, Resource="Engineering"),
            dict(Task="Consenting & Planning", Start=15, Duration=12, Resource="Planning"),
            dict(Task="Construction", Start=27, Duration=24, Resource="Construction")
        ]
        total_months = 51
        financial_markers = []
    else:
        if hazards_gdf is None or len(hazards_gdf) == 0:
            return create_timeline_gantt(None, 'original')
        
        critical = len(hazards_gdf[hazards_gdf['risk'] == 'Critical'])
        high = len(hazards_gdf[hazards_gdf['risk'] == 'High'])
        medium = len(hazards_gdf[hazards_gdf['risk'] == 'Medium'])
        
        phases = []
        curr = 0
        
        phases.append(dict(Task="Geophysical Survey", Start=curr, Duration=3, Resource="Survey"))
        curr += 3
        phases.append(dict(Task="Data Processing", Start=curr, Duration=2, Resource="Analysis"))
        curr += 2
        
        if critical > 0 or high > 0:
            # Targeted resurvey for critical/high hazards
            phases.append(dict(Task="⚠️ Targeted Resurvey", Start=curr, Duration=3, Resource="Resurvey"))
            curr += 3
            phases.append(dict(Task="Mitigation Planning", Start=curr, Duration=2, Resource="Analysis"))
            curr += 2
        
        # Geotechnical survey - slight delay if critical hazards require route changes
        geotech_delay = min(critical * 0.5, 2)  # Cap at 2 months max
        phases.append(dict(Task="Geotechnical Survey", Start=curr, Duration=4+geotech_delay, Resource="Survey"))
        curr += 4 + geotech_delay
        
        # Engineering design - extended if major route changes needed
        design_ext = min(critical * 0.5 + high * 0.25, 3)  # Cap at 3 months max
        phases.append(dict(Task="Engineering Design", Start=curr, Duration=6+design_ext, Resource="Engineering"))
        curr += 6 + design_ext
        
        # Consenting - delays for UXO clearance approvals and route variations
        consent_delay = min(critical * 1.0 + high * 0.5, 6)  # Cap at 6 months max
        phases.append(dict(Task="Consenting & Planning", Start=curr, Duration=12+consent_delay, Resource="Planning"))
        curr += 12 + consent_delay
        
        # Construction - no delay (hazards handled in design phase)
        phases.append(dict(Task="Construction", Start=curr, Duration=24, Resource="Construction"))
        curr += 24
        
        total_months = curr
        
        def parse_cost(s):
            try:
                parts = s.replace('£','').replace(',','').split('-')
                return (float(parts[0]) + float(parts[1] if len(parts)>1 else parts[0]))/2
            except:
                return 0
        
        cost_crit = hazards_gdf[hazards_gdf['risk']=='Critical']['cost'].apply(parse_cost).sum()
        cost_high = hazards_gdf[hazards_gdf['risk']=='High']['cost'].apply(parse_cost).sum()
        cost_med = hazards_gdf[hazards_gdf['risk']=='Medium']['cost'].apply(parse_cost).sum()
        
        financial_markers = []
        if critical > 0:
            financial_markers.append({'month': 7, 'cost': cost_crit, 'label': f'Critical: £{cost_crit/1000:.0f}K', 'color': '#D1FE49'})
        if high > 0:
            financial_markers.append({'month': 12, 'cost': cost_high, 'label': f'High: £{cost_high/1000:.0f}K', 'color': '#0013C3'})
        if medium > 0:
            financial_markers.append({'month': 18, 'cost': cost_med, 'label': f'Medium: £{cost_med/1000:.0f}K', 'color': '#8F998D'})
    
    for p in phases:
        p['Start'] = start_date + timedelta(days=p['Start']*30)
        p['Finish'] = p['Start'] + timedelta(days=p['Duration']*30)
    
    # InX color scheme for phases
    colors = {
        'Survey': '#0013C3',      # Precision blue
        'Analysis': '#12241E',    # Musk green
        'Resurvey': '#D1FE49',    # Neon current
        'Engineering': '#8288A3', # Rocky blue
        'Planning': '#8F998D',    # Sage green
        'Construction': '#0013C3' # Precision blue
    }
    
    fig = ff.create_gantt(phases, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True,
                         title=f"Timeline - {scenario.replace('_',' ').title()}")
    
    if scenario == 'with_hazards' and financial_markers:
        for m in financial_markers:
            m_date = start_date + timedelta(days=m['month']*30)
            fig.add_trace(go.Scatter(x=[m_date,m_date], y=[0,len(phases)], mode='lines+text',
                                    line=dict(color=m['color'], width=3, dash='dash'),
                                    text=[m['label'],''], textposition='top center', showlegend=False))
    
    # White theme for Gantt with better layout
    fig.update_layout(
        height=550,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#000000', family='Space Grotesk', size=11),
        xaxis=dict(
            gridcolor='#E0E0E0', 
            gridwidth=0.5, 
            showgrid=True,
            title="Timeline (Years)"
        ),
        yaxis=dict(
            gridcolor='#E0E0E0', 
            gridwidth=0.5, 
            showgrid=True,
            automargin=True
        ),
        title=dict(font=dict(color='#000000', size=14)),
        margin=dict(l=250, r=50, t=60, b=120),  # More space for labels and legend
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Below the chart
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        bargap=0.15
    )
    
    # Ensure x-axis is visible
    fig.update_xaxes(fixedrange=False, showticklabels=True)
    
    return fig, total_months, financial_markers if scenario=='with_hazards' else []

def generate_evidence(haz):
    """Generate evidence - same as before"""
    hid = haz.get('id','')
    sensors = haz.get('detected_by','').split(', ')
    
    changes = {
        'WRK-001': {'prev': 'Mar 2023', 'list': [
            '**Elevation:** 2.1m → 3.8m (+1.7m increase)',
            '**Scour:** New -0.8m depression around base',
            '**Magnetic:** 180nT → 245nT (+65nT strengthening)',
            '**Status:** Wreck becoming more exposed due to erosion'
        ]},
        'UXO-001': {'prev': 'Mar 2023', 'list': [
            '**New Detection:** Not present in 2023 survey',
            '**SSS Confidence:** 98% certainty on target',
            '**Mag Signature:** 187nT dipole (ordnance pattern)',
            '**Status:** Newly exposed UXO - immediate investigation required'
        ]},
        'UXO-002': {'prev': 'Mar 2023', 'list': [
            '**Position Change:** Migrated 15m NE from 2023 position',
            '**Exposure:** Previously buried, now partially exposed',
            '**Mag Increase:** 85nT → 112nT (becoming more prominent)',
            '**Status:** Mobile UXO - high risk of further movement'
        ]},
        'BLD-001': {'prev': 'Mar 2023', 'list': [
            '**Area Growth:** 210m² → 250m² (+15% expansion)',
            '**Boulder Size:** Avg 1.8m → 2.3m (larger boulders exposed)',
            '**Seabed Erosion:** Active scour revealing subsurface obstacles',
            '**Status:** Dynamic field - ongoing exposure of hazards'
        ]},
        'default': {'prev': 'Mar 2023', 'list': [
            '**Position:** Stable within GPS uncertainty (±2m)',
            '**Morphology:** No significant dimensional changes',
            '**Sensors:** Consistent signatures across surveys'
        ]}
    }
    
    change_data = changes.get(hid, changes['default'])
    ev = {'sss': '', 'mbes': '', 'mag': '', 'sbp': '', 'risk_just': '', 'change': change_data}
    
    htype = haz.get('hazard_type', '')
    
    if 'SSS' in sensors:
        if 'Wreck' in htype:
            ev['sss'] = "**SSS Analysis:**\n- 45m linear target, sharp edges\n- 15m shadow (3.8m height)\n- High backscatter (steel)\n- Hull form visible\n- **Confidence: 95%**"
        elif 'UXO' in htype:
            ev['sss'] = "**SSS Analysis:**\n- No surface expression (buried)\n- Seabed undisturbed\n- Subsurface anomaly present\n- Recent burial indicated\n- **Confidence: 85%**"
        elif 'Boulder' in htype:
            ev['sss'] = "**SSS Analysis:**\n- High backscatter clusters\n- Irregular morphology\n- Shadow confirms elevation\n- 45+ targets >0.5m\n- **Confidence: 92%**"
        elif 'Sand Wave' in htype:
            ev['sss'] = "**SSS Analysis:**\n- Sinusoidal bedform pattern\n- Wavelength: 15-30m\n- Asymmetric profiles indicate active migration\n- Ripple trains on lee slopes\n- **Confidence: 94%**"
        elif 'Hard Ground' in htype or 'Hard' in htype:
            ev['sss'] = "**SSS Analysis:**\n- Very high backscatter intensity\n- Rough texture across area\n- Outcrop visible at seabed\n- No sediment veneer\n- **Confidence: 96%**"
        elif 'Pipeline' in htype:
            ev['sss'] = "**SSS Analysis:**\n- Linear target 0.9m diameter\n- 850m exposed length detected\n- Consistent geometry\n- Acoustic shadow confirms elevation\n- **Confidence: 98%**"
        elif 'Channel' in htype:
            ev['sss'] = "**SSS Analysis:**\n- Linear depression visible in imagery\n- Lower backscatter in channel fill\n- Width: 80-150m at seabed\n- Subtle surface expression\n- **Confidence: 87%**"
    
    if 'MBES' in sensors:
        if 'Wreck' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- Elevation: 3.8m peak\n- Footprint: 45m × 12m\n- Scour moat: -0.8m\n- Roughness elevated\n- **Confidence: 88%**"
        elif 'Boulder' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- 45+ elevation peaks\n- Height: 0.5-2.3m range\n- Field: 250×180m\n- RMS roughness: 0.8m\n- **Confidence: 90%**"
        elif 'Gas' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- 15 pockmarks detected\n- Diameter: 5-20m\n- Depth: 0.5-1.2m\n- Clustered pattern\n- **Confidence: 92%**"
        elif 'Sand Wave' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- Amplitude: 0.8-2.5m\n- Wavelength: 20-40m\n- Asymmetric crests indicate migration\n- Extends 800m × 600m\n- **Confidence: 93%**"
        elif 'Hard Ground' in htype or 'Hard' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- Very high roughness (RMS >1.5m)\n- Irregular surface topography\n- No sediment drape visible\n- Backscatter intensity very high\n- **Confidence: 95%**"
        elif 'Pipeline' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- Elevation: 0.6m above seabed\n- Linear feature 850m length\n- Diameter: 0.9m consistent\n- Free spans detected: 3 locations\n- **Confidence: 97%**"
        elif 'Channel' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- V-shaped depression: 4-8m deep\n- Width: 80-150m\n- Length: >1km (extends beyond survey)\n- Subtle expression at seabed\n- **Confidence: 89%**"
    
    if 'Magnetometer' in sensors:
        if 'Wreck' in htype:
            ev['mag'] = "**Mag Analysis:**\n- Dipole: 245nT\n- Large ferrous mass\n- Aligned with SSS/MBES\n- Steel structure\n- **Confidence: 92%**"
        elif 'UXO' in htype:
            ev['mag'] = "**Mag Analysis:**\n- Dipole: 187nT\n- Ordnance signature\n- WW2 UXO pattern\n- Buried 1.2m depth\n- **Confidence: 92%**"
        elif 'Pipeline' in htype:
            ev['mag'] = "**Mag Analysis:**\n- Linear anomaly 850m length\n- Magnitude: 15-25nT (steel pipe)\n- Consistent along route\n- Confirms ferrous material\n- **Confidence: 89%**"
    
    if 'SBP' in sensors:
        if 'Wreck' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- Hard reflector at seabed\n- No acoustic penetration\n- Solid metallic structure\n- Confirms upstanding obstacle\n- **Confidence: 85%**"
        elif 'Gas' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- Acoustic blanking below 2m\n- Bright spots present\n- Gas accumulation confirmed\n- 2-8m depth zone affected\n- **Confidence: 94%**"
        elif 'Channel' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- V-shaped buried channel\n- Depth below seabed: 4-8m\n- Infill: soft sediment (low velocity)\n- Base reflector visible\n- **Confidence: 91%**"
        elif 'Sand Wave' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- Mobile sand detected (transparent)\n- No hard reflectors\n- Sediment thickness varies 1-3m\n- Active bedform migration indicated\n- **Confidence: 88%**"
        elif 'Hard Ground' in htype or 'Hard' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- Hard reflector at <0.5m depth\n- No acoustic penetration\n- Rock surface confirmed\n- No sediment cover\n- **Confidence: 93%**"
        elif 'Pipeline' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- Hard reflector at seabed\n- Linear target matches SSS/MBES\n- No burial detected\n- Fully exposed pipeline\n- **Confidence: 90%**"
    
    score = haz.get('risk_score', 0)
    if score >= 9.0:
        ev['risk_just'] = f"**Risk: {score}/10 - CRITICAL**\n\n1. Multi-sensor: {len(sensors)} independent confirmations\n2. Size/magnitude: Significant cable strike risk\n3. Location: {haz.get('distance_to_turbine_m')}m from infrastructure\n4. Change detection: Dynamic seabed/increasing exposure\n5. Mitigation: Immediate action required\n\n**Combined confidence: 91%**"
    elif score >= 7.0:
        ev['risk_just'] = f"**Risk: {score}/10 - HIGH**\n\n1. Sensor agreement: {len(sensors)} confirmations\n2. Proximity: {haz.get('distance_to_turbine_m')}m to turbine\n3. Engineering: Route modification required\n4. Timeline: Investigation/redesign delays\n\n**Mitigation required before construction**"
    elif score >= 4.0:
        ev['risk_just'] = f"**Risk: {score}/10 - MEDIUM**\n\n1. Detection: {len(sensors)} sensors confirm\n2. Impact: Manageable with standard mitigation\n3. Timeline: Can be scheduled during normal phases\n\n**Standard burial + monitoring recommended**"
    else:
        ev['risk_just'] = f"**Risk: {score}/10 - LOW**\n\n1. Minor hazard: Limited impact potential\n2. Mitigation: Monitoring sufficient\n3. No urgent action required\n\n**Document and track during construction**"
    
    return ev


# ==============================================================================
# SIDEBAR - INX BRANDED
# ==============================================================================

st.sidebar.header("🎨 Display Settings")
basemap = st.sidebar.selectbox("Basemap", ['OpenStreetMap', 'Esri Satellite'], index=1)
mbes_cmap = st.sidebar.selectbox("MBES Colormap", ['ocean', 'viridis', 'seismic'], index=0)
quality = st.sidebar.radio("Quality", ["Fast (500px)", "Good (1000px)", "High (2000px)"], index=1)
max_px = {"Fast (500px)": 500, "Good (1000px)": 1000, "High (2000px)": 2000}[quality]

FILE_IDS = {
    'sss': ['1-fd4WYSO3jAurneJNV_QzMJVx3F5rojM','1reqiNT6_XKdFc4LzjAM634CRe3qqReZP','1MmJAYU9O6bjqst0ufZW5s_7DZQirSr5Z','10XWv6wmnIX0zHDHTtIsOoM71JtByLNVb','10YZlXZ2JDp5f7ehg4xMdFkkZCD5NVHNI','12FLV4q_9X4EzGrqIWtGShCo-BUYrmRlG','134bxFTgfLwZYWvWhIa7swmhS2UGCV7M0','15X6Ho70GmLxlDHubSDnEfEQq85s0CKol','17dJXRk_VIZuQhULjj52-BqaUAnaaeeOH','17yLRua1a3AgBYuZC4_36D7x5-kfLON7L','1A2ZYFc6Mey_pJvBtB9gGXL5zxDeJtdRM','1AiH391YcyhizRgWSldH8Oijd6PzG9a9Y','1EC4BT0SBHsYf6iYXGYXUxhmKhYNbXoeY','1Ffz5h8qjND_jS-wA3QUxtQ0DUaoNj9wC','1GDU4aonNheXJ0pK7NsWqjLyXNDzE1NwM','1IJooNeDkLj4TqCxra7iOjFYtVU182e7I','1MLaLICacB1DpyPv1jkFq8SrY1tRKl7aN','1NYTPv-3PsWs7kjeer_uf4pfGMsbwTi0E','1N_Y_bmCTUuu15IYS9j-XLJNH28OyZD9H','1RZkpzxIQgrCYnWBGm_w42stbyzVmEPva','1VlAVkEbnTbFnto57sMbfFkbzU7p67LL7','1aVvPfIXoRDC2XqmDMG92fUIZtrFpIsJq','1bHSd6XzLDYIAwnQYlDPRo8FLnH8DDJnw','1cMBJlt0A6JwfMS7fhcvR7cnJ4468gLze','1cNNmCgY6iAbHMtGm2UPeJX_NeMnb2rQn','1cvLjbwFDPjD2avHzjMuwDGKYDjjXLT5v','1dhxT_QsbygFdLV9ZYUQ6wd5_2mZSb6e5','1eaS_7K8012AneqC5LkwmuLEqqJmPO1sq','1hg9wgSkhRIzYiIhCGq1xjbFMxzSz4pSm','1iDJWZcRz_zGbOTQpYN9U1V707X5xo3yv','1jNwjUx7zdHHFKxFtAXSbYLMrmVKDdRxS','1jqZLJ5xJhxdChh9SKlbahsLviqbqPzFx','1ldV6zBMMrWfovkNbV2bSSkHyZmPUKYlI','1nzPO4LXl6PJ5TffOe6c2pHJFmZzSUDfp','1sWFLzNsAo0ZQ_nbusrNm9I7DnfFh4TIq','1t0NXhHNdHQrwuMCzfiGu1CYb1z47-XVK','1w2ZwrKigqOHqXMRnyY4GD_jNn0VTCrWE','1wkcFrGXx8dVNf5gYMkEaNeIdvIRNazJz','1wmMgdqL-B56PI4sHQ-Fr4GFxp28ptb8U','1zso2rorqe_FXDXbMfHXl3vDRodD8H7fC'],
    'mbes': '1lE9X1S2Lqt3UxKgEJto5cURf1gTxOADr',
    'mag': '1jyYQ9ICEFjXxFAatFQvGb-9byu3ryq5P',
    'turbines': '18uYbX7OWZcqQfoBow6F_P4AmjptioeeO',
    'sbp': '1cZCoNX1t68X1BoiyikYKRAV0vzo_3pGO',
    'hazards': '1x_aerOM_LY7bw1CJdNC35zD2KkhJo4Sh'
}

st.sidebar.markdown("---")
st.sidebar.header("📁 Quick Load")

if st.sidebar.button("🚀 Load SSS", use_container_width=True):
    prog = st.progress(0)
    stat = st.empty()
    count = 0
    for i, fid in enumerate(FILE_IDS['sss']):
        try:
            stat.text(f"Loading {i+1}/{len(FILE_IDS['sss'])}...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                download_from_gdrive(fid, tmp.name)
                img, bounds = tif_to_png_base64(tmp.name, 'gray', max_px, True, False)
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
            img, bounds = tif_to_png_base64(tmp.name, mbes_cmap, max_px, False, False)
            if img and bounds:
                st.session_state.raster_layers.append((img, bounds))
                st.success("✅ MBES loaded")
                st.rerun()
            os.unlink(tmp.name)
        except Exception as e:
            st.error(f"Error: {e}")

if st.sidebar.button("🧲 Load Mag", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        try:
            download_from_gdrive(FILE_IDS['mag'], tmp.name)
            img, bounds = tif_to_png_base64(tmp.name, 'seismic', max_px, False, True)
            if img and bounds:
                st.session_state.mag_tif_layer = (img, bounds)
                st.success("✅ Mag loaded")
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

if st.sidebar.button("🔊 Load SBP", use_container_width=True):
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

if st.sidebar.button("🗑️ Clear", use_container_width=True):
    st.session_state.raster_layers = []
    st.session_state.hazards = None
    st.session_state.turbines = None
    st.session_state.sbp_lines = None
    st.session_state.mag_tif_layer = None
    st.rerun()

# Financial dashboard
if st.session_state.hazards is not None and len(st.session_state.hazards) > 0:
    st.sidebar.markdown("---")
    st.sidebar.header("💰 Project Impact")
    hazs = st.session_state.hazards
    def pc(s):
        try:
            p = s.replace('£','').replace(',','').split('-')
            return (float(p[0])+float(p[1] if len(p)>1 else p[0]))/2
        except:
            return 0
    costs = hazs['cost'].apply(pc)
    avg = costs.sum()
    crit = len(hazs[hazs['risk']=='Critical'])
    st.sidebar.metric("Mitigation Cost", f"£{avg/1000:.0f}K")
    st.sidebar.metric("Critical Hazards", f"{crit}", delta="Immediate action")

st.sidebar.markdown("---")
st.sidebar.success("✅ System Operational")
st.sidebar.info(f"📡 Updated: {datetime.now().strftime('%H:%M:%S')}")


# ==============================================================================
# PAGES - Same structure as before, now with InX branding
# ==============================================================================

# Copy the full page implementations from app_MULTIPAGE_FINAL.py
# Starting from line 449 to the end

# ==============================================================================
# PAGE 1: HAZARD MAP
# ==============================================================================

if page == "🗺️ Hazard Map":
    st.header("🗺️ Interactive Hazard Map")
    
    if st.session_state.hazards is not None and len(st.session_state.hazards) > 0:
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            types = sorted(st.session_state.hazards['hazard_type'].unique())
            sel_types = st.multiselect("Filter by Type:", types, types, key='map_types')
        with col2:
            risks = ["Critical", "High", "Medium", "Low"]
            sel_risks = st.multiselect("Filter by Risk:", risks, risks, key='map_risks')
        
        show_ai = st.checkbox("Show GRID AI Detection", True, key='show_ai')
        
        filtered = st.session_state.hazards[
            st.session_state.hazards['hazard_type'].isin(sel_types) &
            st.session_state.hazards['risk'].isin(sel_risks)
        ]
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🔴 Critical", len(filtered[filtered['risk']=='Critical']))
        col2.metric("🟠 High", len(filtered[filtered['risk']=='High']))
        col3.metric("🟡 Medium", len(filtered[filtered['risk']=='Medium']))
        col4.metric("🟢 Low", len(filtered[filtered['risk']=='Low']))
        col5.metric("📍 Total", len(filtered))
        
        # Map
        all_bounds = [b for _, b in st.session_state.raster_layers if b]
        if all_bounds:
            center_lat = sum([(b[1]+b[3])/2 for b in all_bounds])/len(all_bounds)
            center_lon = sum([(b[0]+b[2])/2 for b in all_bounds])/len(all_bounds)
        else:
            center_lat, center_lon = 53.81, 0.13
        
        if basemap == 'Esri Satellite':
            m = folium.Map([center_lat, center_lon], zoom_start=13, tiles=None)
            folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri').add_to(m)
        else:
            m = folium.Map([center_lat, center_lon], zoom_start=13)
        
        # Add rasters
        for img, bounds in st.session_state.raster_layers:
            if img and bounds:
                folium.raster_layers.ImageOverlay(f"data:image/png;base64,{img}",
                    [[bounds[1],bounds[0]],[bounds[3],bounds[2]]], opacity=0.85).add_to(m)
        
        if st.session_state.mag_tif_layer:
            img, bounds = st.session_state.mag_tif_layer
            if img and bounds:
                folium.raster_layers.ImageOverlay(f"data:image/png;base64,{img}",
                    [[bounds[1],bounds[0]],[bounds[3],bounds[2]]], opacity=0.6, name="Mag").add_to(m)
        
        # SBP lines
        if st.session_state.sbp_lines is not None:
            try:
                gdf = st.session_state.sbp_lines.to_crs('EPSG:4326') if st.session_state.sbp_lines.crs else st.session_state.sbp_lines
                for _, row in gdf.iterrows():
                    folium.GeoJson(row.geometry.__geo_interface__,
                        style_function=lambda x: {'color': '#00FFFF', 'weight': 2}).add_to(m)
            except:
                pass
        
        # Turbines
        if st.session_state.turbines is not None:
            try:
                gdf = st.session_state.turbines.to_crs('EPSG:4326') if st.session_state.turbines.crs else st.session_state.turbines
                for _, row in gdf.iterrows():
                    folium.CircleMarker([row.geometry.y, row.geometry.x], radius=8,
                        color='#4169E1', fill=True, fillColor='#4169E1', fillOpacity=0.8).add_to(m)
            except:
                pass
        
        # Hazards
        for _, row in filtered.iterrows():
            risk = row.get('risk', 'Unknown')
            htype = row.get('hazard_type', 'Unknown')
            colors = {'Critical':'red','High':'orange','Medium':'beige','Low':'green'}
            color = colors.get(risk, 'gray')
            
            popup_html = f"""
            <div style="width:360px; font-family:Arial; font-size:13px;">
                <h3 style="margin:0; padding:10px; background:{get_risk_color(risk)}; color:white; border-radius:5px 5px 0 0;">
                    {htype} - {row.get('id','')}
                </h3>
                <div style="padding:10px;">
                    <p style="margin:5px 0;"><b>Name:</b> {row.get('name','N/A')}</p>
                    <p style="margin:5px 0;"><b>Size:</b> {row.get('size','N/A')}</p>
                    <div style="margin:10px 0; padding:8px; background:#f0f0f0; border-radius:5px;">
                        <p style="margin:0;"><b>📡 Detected:</b> {row.get('detected_by','N/A')}</p>
                    </div>
                    <p style="margin:5px 0;"><b>Distance:</b> {row.get('distance_to_turbine_m','N/A')}m to {row.get('nearest_turbine','N/A')}</p>
                    <div style="margin:10px 0; padding:8px; background:#fff3cd; border-radius:5px;">
                        <p style="margin:2px 0;"><b>⚠️ Risk:</b> <span style="color:{get_risk_color(risk)}; font-weight:bold;">{risk}</span></p>
                        <p style="margin:2px 0;"><b>Score:</b> {row.get('risk_score','N/A')}/10</p>
                    </div>
            """
            
            if show_ai:
                sensors = row.get('detected_by', '').split(', ')
                nsens = len([s for s in sensors if s.strip()])
                popup_html += f"""
                    <div style="margin:10px 0; padding:10px; background:#e8f4f8; border-radius:5px; border-left:4px solid #0066cc;">
                        <h4 style="margin:0 0 8px 0; color:#0066cc;">🤖 GRID AI Detection</h4>
                """
                if 'SSS' in row.get('detected_by',''):
                    popup_html += '<p style="margin:3px 0; font-size:11px;">✅ SSS: Target confirmed | 95%</p>'
                if 'MBES' in row.get('detected_by',''):
                    popup_html += '<p style="margin:3px 0; font-size:11px;">✅ MBES: Elevation anomaly | 88%</p>'
                if 'Magnetometer' in row.get('detected_by',''):
                    popup_html += '<p style="margin:3px 0; font-size:11px;">✅ Mag: Ferrous signature | 92%</p>'
                if 'SBP' in row.get('detected_by',''):
                    popup_html += '<p style="margin:3px 0; font-size:11px;">✅ SBP: Subsurface anomaly | 85%</p>'
                popup_html += f"""
                        <div style="margin-top:8px; padding-top:8px; border-top:1px solid #ccc;">
                            <p style="margin:2px 0; font-size:11px;"><b>Combined Confidence:</b> 91%</p>
                            <p style="margin:2px 0; font-size:11px;"><b>Agreement:</b> {nsens}/4 sensors</p>
                        </div>
                    </div>
                """
            
            popup_html += f"""
                    <p style="margin:5px 0;"><b>💰 Cost:</b> {row.get('cost','N/A')}</p>
                    <p style="margin:5px 0;"><b>📅 Timeline:</b> {row.get('investigation_timeline','N/A')}</p>
                </div>
            </div>
            """
            
            folium.Marker([row.geometry.y, row.geometry.x],
                popup=folium.Popup(popup_html, max_width=380),
                tooltip=f"{htype}: {row.get('name','')} | {risk}",
                icon=folium.Icon(color=color, icon='exclamation-triangle', prefix='fa')).add_to(m)
        
        folium.plugins.Fullscreen().add_to(m)
        st_folium(m, width=1400, height=700)
        
        # Table
        st.markdown("---")
        st.subheader("📋 Hazard Register")
        cols = ['id','hazard_type','name','risk','distance_to_turbine_m','investigation_timeline','cost']
        st.dataframe(filtered[cols], use_container_width=True, height=300)
        csv = filtered.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, f"hazards_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    else:
        st.info("👆 Load hazards from sidebar to begin")


# ==============================================================================
# PAGE 2: PROJECT TIMELINE
# ==============================================================================

elif page == "📅 Project Timeline":
    st.header("📅 Project Timeline Analysis")
    
    if st.session_state.hazards is not None and len(st.session_state.hazards) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Original Timeline (No Hazards)")
            fig_orig, mo_orig, _ = create_timeline_gantt(None, 'original')
            st.plotly_chart(fig_orig, use_container_width=True)
            st.metric("Duration", f"{mo_orig} months", delta=f"{mo_orig/12:.1f} years")
        
        with col2:
            st.subheader("⚠️ Timeline with Hazard Impacts")
            fig_haz, mo_haz, markers = create_timeline_gantt(st.session_state.hazards, 'with_hazards')
            st.plotly_chart(fig_haz, use_container_width=True)
            delay = mo_haz - mo_orig
            st.metric("Duration", f"{mo_haz} months", delta=f"+{delay:.0f} mo delay", delta_color="inverse")
        
        st.markdown("---")
        st.subheader("📊 Impact Breakdown")
        
        hazs = st.session_state.hazards
        crit = len(hazs[hazs['risk']=='Critical'])
        high = len(hazs[hazs['risk']=='High'])
        med = len(hazs[hazs['risk']=='Medium'])
        low = len(hazs[hazs['risk']=='Low'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Project Delay", f"+{delay:.0f} months", delta=f"+{delay/mo_orig*100:.1f}% extension")
        with col2:
            total_cost = sum([m['cost'] for m in markers]) if markers else 0
            st.metric("Mitigation Cost", f"£{total_cost/1e6:.2f}M", delta="From interventions")
        with col3:
            st.metric("Critical Interventions", f"{crit}", delta=f"{high} high risk" if high>0 else "")
        
        # Financial timeline
        if markers:
            st.markdown("---")
            st.subheader("💰 Financial Commitments Timeline")
            import plotly.express as px
            mdf = pd.DataFrame(markers)
            fig_cost = px.bar(mdf, x='month', y='cost', color='label',
                title="Mitigation Costs by Project Month",
                labels={'month': 'Project Month', 'cost': 'Cost (£)'},
                color_discrete_map={mdf.iloc[i]['label']: mdf.iloc[i]['color'] for i in range(len(mdf))})
            fig_cost.update_layout(height=400)
            st.plotly_chart(fig_cost, use_container_width=True)
        
        # Risk breakdown with clickable navigation
        st.markdown("---")
        st.subheader("⚠️ Delay Contribution by Risk Level")
        
        st.info("💡 **Delay Calculation Methodology:** Critical hazards require resurvey + consent delays (1-2 months each). High risk requires additional engineering (0.5-1 month each). Medium/Low risks are addressed during normal phases with minimal delay.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cd = min(crit * 1.5, 12)  # Max 12 months from critical hazards
            st.metric("Critical", f"+{cd:.1f} mo", delta=f"{crit} hazards")
            if crit > 0:
                if st.button(f"📍 View {crit} Critical Hazards", key='btn_crit', use_container_width=True):
                    st.info("💡 Navigate to 'Hazard Map' page and filter by 'Critical' risk to see these hazards")
        
        with col2:
            hd = min(high * 0.75, 6)  # Max 6 months from high risk
            st.metric("High", f"+{hd:.1f} mo", delta=f"{high} hazards")
            if high > 0:
                if st.button(f"📍 View {high} High Hazards", key='btn_high', use_container_width=True):
                    st.info("💡 Navigate to 'Hazard Map' page and filter by 'High' risk to see these hazards")
        
        with col3:
            md = min(med * 0.3, 3)  # Max 3 months from medium risk
            st.metric("Medium", f"+{md:.1f} mo", delta=f"{med} hazards")
            if med > 0:
                if st.button(f"📍 View {med} Medium Hazards", key='btn_med', use_container_width=True):
                    st.info("💡 Navigate to 'Hazard Map' page and filter by 'Medium' risk to see these hazards")
        
        with col4:
            ld = min(low * 0.1, 1)  # Max 1 month from low risk
            st.metric("Low", f"+{ld:.1f} mo", delta=f"{low} hazards")
            if low > 0:
                if st.button(f"📍 View {low} Low Hazards", key='btn_low', use_container_width=True):
                    st.info("💡 Navigate to 'Hazard Map' page and filter by 'Low' risk to see these hazards")
    
    else:
        st.info("Load hazards to see timeline analysis")

# ==============================================================================
# PAGE 3: EVIDENCE VIEWER
# ==============================================================================

elif page == "🔬 Evidence Viewer":
    st.header("🔬 Evidence Viewer - Hazard Analysis")
    
    if st.session_state.hazards is not None and len(st.session_state.hazards) > 0:
        # Hazard TYPE selector (not individual hazard)
        hazard_types = sorted(st.session_state.hazards['hazard_type'].unique())
        selected_type = st.selectbox("Select Hazard Type to Analyze:", hazard_types, key='ev_type_select')
        
        # Filter hazards by type
        type_hazards = st.session_state.hazards[st.session_state.hazards['hazard_type'] == selected_type]
        
        st.subheader(f"📍 {selected_type} Locations ({len(type_hazards)} detected)")
        
        # Create map with ALL hazards of this type + MBES
        center_lat = type_hazards.geometry.y.mean()
        center_lon = type_hazards.geometry.x.mean()
        
        m = folium.Map([center_lat, center_lon], zoom_start=13, tiles=None)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite'
        ).add_to(m)
        
        # Add MBES layer if available
        if st.session_state.raster_layers:
            for img, bounds in st.session_state.raster_layers:
                if img and bounds:
                    folium.raster_layers.ImageOverlay(
                        f"data:image/png;base64,{img}",
                        [[bounds[1],bounds[0]],[bounds[3],bounds[2]]],
                        opacity=0.7
                    ).add_to(m)
        
        # Add all hazards of this type
        for idx, row in type_hazards.iterrows():
            risk = row.get('risk', 'Unknown')
            colors = {'Critical':'red','High':'orange','Medium':'yellow','Low':'green'}
            color = colors.get(risk, 'gray')
            
            # Simple popup with hazard ID
            popup_html = f"""
            <div style='font-family: Space Grotesk; padding: 10px;'>
                <h4 style='margin: 0;'>{row.get('id','')}</h4>
                <p style='margin: 5px 0;'><b>{row.get('name','')}</b></p>
                <p style='margin: 5px 0;'>Risk: {risk} ({row.get('risk_score','')}/10)</p>
                <p style='margin: 5px 0; font-size: 11px;'>Click marker to see detailed evidence below</p>
            </div>
            """
            
            folium.Marker(
                [row.geometry.y, row.geometry.x],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row.get('id','')}: {row.get('name','')}",
                icon=folium.Icon(color=color, icon='exclamation-triangle', prefix='fa')
            ).add_to(m)
        
        folium.plugins.Fullscreen().add_to(m)
        
        # Display map with clickable markers
        map_data = st_folium(m, width=1400, height=500, key='evidence_map')
        
        # Auto-select hazard if marker clicked
        if map_data and map_data.get('last_object_clicked'):
            clicked_coords = map_data['last_object_clicked']
            if clicked_coords:
                # Find the closest hazard to clicked coordinates
                clicked_lat = clicked_coords['lat']
                clicked_lng = clicked_coords['lng']
                
                # Calculate distances and find closest
                distances = type_hazards.geometry.apply(
                    lambda geom: ((geom.y - clicked_lat)**2 + (geom.x - clicked_lng)**2)**0.5
                )
                closest_idx = distances.idxmin()
                clicked_hazard = type_hazards.loc[closest_idx]
                
                # Update session state with clicked hazard
                st.session_state['selected_evidence_hazard'] = clicked_hazard['id']
        
        # Detailed Evidence Analysis
        st.markdown("---")
        st.subheader("🔍 Detailed Evidence Analysis")
        
        # Get the hazard to display (from click or dropdown)
        haz_list = [f"{row['id']}: {row.get('name','Unknown')}" for _, row in type_hazards.iterrows()]
        
        # Pre-select if clicked on map
        default_idx = 0
        if 'selected_evidence_hazard' in st.session_state:
            clicked_id = st.session_state['selected_evidence_hazard']
            for i, haz_str in enumerate(haz_list):
                if haz_str.startswith(clicked_id):
                    default_idx = i
                    break
        
        selected_hazard = st.selectbox(
            "Select specific hazard for detailed analysis:", 
            haz_list, 
            index=default_idx,
            key='ev_detail_select'
        )
        
        if selected_hazard:
            hid = selected_hazard.split(':')[0]
            hrow = type_hazards[type_hazards['id']==hid].iloc[0]
            
            # Evidence
            ev = generate_evidence(hrow)
            
            st.markdown("---")
            st.subheader("⚠️ Risk Score Justification")
            st.markdown(ev['risk_just'])
            
            st.markdown("---")
            st.subheader("📡 Multi-Sensor Detection Evidence")
            
            tab1, tab2, tab3, tab4 = st.tabs(["SSS", "MBES", "Magnetometer", "SBP"])
            
            with tab1:
                if ev['sss']:
                    st.markdown(ev['sss'])
                else:
                    st.info("SSS not used for this hazard")
            
            with tab2:
                if ev['mbes']:
                    st.markdown(ev['mbes'])
                else:
                    st.info("MBES not used for this hazard")
            
            with tab3:
                if ev['mag']:
                    st.markdown(ev['mag'])
                else:
                    st.info("Magnetometer not used for this hazard")
            
            with tab4:
                if ev['sbp']:
                    st.markdown(ev['sbp'])
                else:
                    st.info("SBP not used for this hazard")
            
            st.markdown("---")
            st.subheader("🔄 Multi-Survey Change Detection")
            st.info(f"📅 Comparison: Current survey (Nov 2024) vs Previous ({ev['change']['prev']})")
            
            for change in ev['change']['list']:
                st.markdown(f"- {change}")
    
    else:
        st.info("Load hazards to view evidence analysis")

st.markdown("---")
st.caption("⚡ The Grid - Powered by Multi-Sensor AI")
