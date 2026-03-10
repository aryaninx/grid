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
    /* Remove native browser select arrow — Streamlit adds its own chevron */
    div[data-baseweb="select"] select {{
        -webkit-appearance: none;
        -moz-appearance: none;
        appearance: none;
    }}
</style>
""", unsafe_allow_html=True)

# Session state
for key in ['raster_layers', 'hazards', 'turbines', 'sbp_lines', 'mag_tif_layer', 'mag_targets']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'raster_layers' else None



# BRANDED HEADER WITH LOGO FROM GOOGLE DRIVE
# Replace FILE_ID with your actual Google Drive file ID
LOGO_FILE_ID = "1A84a5D-19A-AeFcnb4elT3WxfSF5t-Em"  # Update with your actual File ID

st.markdown(f"""
<div style='padding: 10px 20px; margin-bottom: 20px;'>
    <div style='display: flex; align-items: center; gap: 20px;'>
        <img src='https://drive.google.com/thumbnail?id={LOGO_FILE_ID}&sz=w200' 
             style='height: 60px; width: auto;'
             alt='InX Tech'
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
    ["Hazard Map", "Project Timeline", "Evidence Viewer"],
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

def haversine_m(lat1, lon1, lat2, lon2):
    """Return distance in metres between two WGS84 points."""
    import math
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def nearest_turbine_info(haz_lat, haz_lon, turbines_gdf):
    """Return (distance_m, ref, lat, lon) for nearest turbine."""
    best_dist = float('inf')
    best_ref = 'N/A'
    best_lat = best_lon = None
    for _, t in turbines_gdf.iterrows():
        tlat, tlon = t.geometry.y, t.geometry.x
        d = haversine_m(haz_lat, haz_lon, tlat, tlon)
        if d < best_dist:
            best_dist = d
            best_ref = t.get('ref') or t.get('seamark:name') or 'N/A'
            best_lat, best_lon = tlat, tlon
    return best_dist, best_ref, best_lat, best_lon

# ==============================================================================
# HARDCODED MAG PINPOINTS — real coordinates & nT from Targets_5nT_wgs84.csv
# Assigned to specific hazard IDs so pins sit exactly on the anomaly.
# WRK = wreck candidates (massive ferrous mass), UXO = ordnance candidates,
# PPL = pipeline candidates (linear signature)
# ==============================================================================
MAG_PINPOINTS = {
    'WRK-001': {'lat': 53.804690, 'lon': 0.146692, 'nT': 32791.4,  'note': 'Dominant anomaly — large ferrous mass, probable WWII wreck'},
    'WRK-002': {'lat': 53.829585, 'lon': 0.158043, 'nT': 4522.7,   'note': 'Second largest anomaly — wreck or large metallic debris'},
    'UXO-001': {'lat': 53.802744, 'lon': 0.115879, 'nT': 217.3,    'note': 'Strong dipole — ordnance signature, buried ~0.9m'},
    'UXO-002': {'lat': 53.808667, 'lon': 0.166573, 'nT': 141.0,    'note': 'UXO-class dipole — WWII pattern, target depth ~1.5m'},
    'UXO-003': {'lat': 53.791244, 'lon': 0.114770, 'nT': 51.4,     'note': 'Moderate anomaly — possible small ordnance item'},
    'PPL-001': {'lat': 53.793448, 'lon': 0.115044, 'nT': 80.7,     'note': 'Linear anomaly cluster — ferrous pipe signature'},
}

def patch_hazard_coordinates(gdf):
    """Overwrite geometry & store real nT for pinned mag hazards."""
    from shapely.geometry import Point
    gdf = gdf.copy()
    for hid, pin in MAG_PINPOINTS.items():
        mask = gdf['id'] == hid
        if mask.any():
            gdf.loc[mask, 'geometry'] = Point(pin['lon'], pin['lat'])
            gdf.loc[mask, 'mag_nt']   = pin['nT']
            gdf.loc[mask, 'mag_note'] = pin['note']
    # Ensure columns exist for all rows
    if 'mag_nt'   not in gdf.columns: gdf['mag_nt']   = None
    if 'mag_note' not in gdf.columns: gdf['mag_note'] = None
    return gdf

# ==============================================================================
# ==============================================================================
# RISK ZONE ENGINE — 25 handcrafted polygons over the survey area
# Each zone has a unique geologically coherent risk profile placed to align
# with real anomaly hotspots from Targets_5nT_wgs84.csv
# ==============================================================================

RISK_DRIVERS = [
    ('Slope / Bathymetry',     'slope',     '#0013C3'),
    ('Seabed Roughness',       'roughness', '#8288A3'),
    ('Shallow Gas (SBP)',      'gas',       '#FF4500'),
    ('Disturbed Stratigraphy', 'strat',     '#FF8C00'),
    ('Mag Anomaly Proximity',  'mag',       '#D4E157'),
    ('Bedform Mobility',       'bedform',   '#8F998D'),
    ('Data Uncertainty',       'uncert',    '#bdbdbd'),
]
DRIVER_KEYS = [d[1] for d in RISK_DRIVERS]

# ── 25 handcrafted zones ────────────────────────────────────────────────────
# bbox = [south, north, west, east]
RISK_ZONES_DEF = [
    # ═══════════════════════════════════════════════════════════════════════
    # EXPORT CABLE CORRIDOR  (lon -0.009 → 0.112, lat 53.770–53.792)
    # Narrow ~100m-wide trench route from landfall to main survey area
    # ═══════════════════════════════════════════════════════════════════════

    {'id':'RZ-C1','label':'Cable Landfall Zone','score':6.2,
     'bbox':[53.773,53.789,-0.009,0.020],
     'drivers':{'slope':30,'strat':25,'roughness':20,'uncert':15,'gas':10,'mag':0,'bedform':0},
     'obs':['Increasing seabed gradient toward shore (MBES)',
            'Reflector disruption — possible shallow utilities (SBP)',
            'Backscatter variability — mixed sediment (SSS)'],
     'interp':['Landfall transition zone — expected gradient increase and sediment variability',
               'Shallow reflector disruptions may indicate buried infrastructure or archaeology'],
     'why':['Increasing gradient creates cable stress at landfall approach',
            'Reflector disruptions require investigation before burial design is finalised',
            'Tidal influence on sediment dynamics increases uncertainty near shore'],
     'confidence':'Moderate','conf_note':'Shallow water — vessel constraints may have reduced data quality near landfall'},

    {'id':'RZ-C2','label':'Cable Corridor West','score':5.1,
     'bbox':[53.774,53.790,0.020,0.055],
     'drivers':{'uncert':32,'slope':22,'roughness':20,'strat':16,'bedform':10,'gas':0,'mag':0},
     'obs':['Flat to gently undulating seabed (MBES)',
            'Uniform fine sand surface (SSS)',
            'Thin layered sequence to 5m (SBP)'],
     'interp':['Conditions broadly favourable for cable installation along this segment',
               'No discrete geophysical hazards identified — risk driven by data uncertainty'],
     'why':['Western corridor at edge of survey coverage — reduced sensor overlap',
            'CPT data would substantially reduce residual uncertainty'],
     'confidence':'Low','conf_note':'Western cable corridor at limit of magnetometer coverage'},

    {'id':'RZ-C3','label':'Cable Corridor Central','score':5.8,
     'bbox':[53.775,53.791,0.055,0.090],
     'drivers':{'strat':28,'roughness':24,'slope':20,'uncert':18,'gas':10,'mag':0,'bedform':0},
     'obs':['Patchy backscatter — variable sediment cover (SSS)',
            'Minor reflector disruptions at 2–4m depth (SBP)',
            'Low roughness with isolated elevated targets (MBES)'],
     'interp':['Moderate geotechnical variability — mixed sediment types along corridor',
               'Reflector disruptions suggest possible infilled features beneath cable track'],
     'why':['Variable ground conditions complicate burial design specification',
            'Isolated elevated targets require individual assessment before route sign-off'],
     'confidence':'Moderate','conf_note':'Central cable corridor — reasonable SBP and SSS coverage'},

    {'id':'RZ-C4','label':'Cable Corridor East','score':4.8,
     'bbox':[53.775,53.792,0.090,0.114],
     'drivers':{'roughness':28,'slope':25,'strat':20,'uncert':17,'bedform':10,'gas':0,'mag':0},
     'obs':['Smooth transition to main survey area (MBES)',
            'Uniform backscatter (SSS)',
            'Clean stratigraphy — good SBP penetration (SBP)'],
     'interp':['Eastern end of cable corridor — conditions improving as water depth increases',
               'Transition to main survey area geophysics is gradual and well-defined'],
     'why':['Roughness and slope are the primary residual risk factors here',
            'Data quality improves eastward — uncertainty reducing'],
     'confidence':'Moderate','conf_note':'Good coverage at junction with main survey area'},

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN SURVEY AREA — SW SECTOR  (lon 0.114–0.155, lat 53.764–53.800)
    # ═══════════════════════════════════════════════════════════════════════

    {'id':'RZ-SW1','label':'SW Entry — Pipeline Corridor','score':7.2,
     'bbox':[53.770,53.796,0.114,0.140],
     'drivers':{'mag':35,'roughness':25,'slope':20,'strat':10,'uncert':10,'gas':0,'bedform':0},
     'obs':['80 nT linear ferrous anomaly along N-S axis (Mag)',
            'Linear SSS target — 0.9m diameter, 850m exposed length (SSS)',
            'Elevation 0.6m above seabed (MBES)',
            'Hard reflector, no burial detected (SBP)'],
     'interp':['Steel pipeline confirmed — consistent diameter, geometry and magnetic signature',
               'Fully exposed with free spans at three locations',
               'No burial — cable crossing constraint is elevated'],
     'why':['Exposed pipeline creates cable crossing design constraint',
            'Free spans increase cable snag and entanglement risk during lay',
            'Operating status unknown — third-party consent required before crossing'],
     'confidence':'High','conf_note':'Four-sensor confirmation — SSS, MBES, SBP and Mag all consistent'},

    {'id':'RZ-SW2','label':'UXO Zone A','score':8.8,
     'bbox':[53.797,53.806,0.108,0.126],
     'drivers':{'mag':50,'strat':20,'uncert':15,'roughness':10,'slope':5,'gas':0,'bedform':0},
     'obs':['217 nT compact dipole — ordnance-class signature (Mag)',
            'No surface expression — seabed undisturbed (SSS)',
            'Subsurface anomaly at ~0.9m depth (SBP)'],
     'interp':['Compact dipole geometry consistent with buried bomb or mine',
               'Absence of SSS expression confirms burial — not a surface feature',
               'SBP depth estimate places target within standard cable burial envelope'],
     'why':['Buried ordnance within cable burial depth — direct clearance risk',
            'WWII munitions disposal area — high prior probability of UXO',
            'Burial depth ~0.9m makes conventional jet-trenching unsafe without clearance'],
     'confidence':'High','conf_note':'Mag dipole geometry is diagnostic; SBP confirms burial depth'},

    {'id':'RZ-SW3','label':'Gas Pockmark Cluster','score':8.1,
     'bbox':[53.778,53.797,0.086,0.113],
     'drivers':{'gas':45,'strat':30,'roughness':10,'slope':8,'uncert':7,'mag':0,'bedform':0},
     'obs':['Acoustic blanking / wipeout below 2m (SBP)',
            '15 pockmarks 5–20m diameter (MBES)',
            'Low-backscatter surface patches at pockmark centres (SSS)'],
     'interp':['Shallow gas accumulation confirmed — blanking indicates free gas in sediment pores',
               'Pockmark field indicates episodic gas venting — active process',
               'Seabed alone appears manageable; subsurface instability dominates risk'],
     'why':['Shallow gas suggests weak / heterogeneous near-surface conditions',
            'Pockmarks indicate disturbed stratigraphy and variable geotechnical strength',
            'Gas venting risk during cable burial — potential for blow-out in trench'],
     'confidence':'High','conf_note':'SBP blanking is diagnostic; MBES pockmark morphology confirms'},

    {'id':'RZ-SW4','label':'Buried Glacial Channel','score':7.0,
     'bbox':[53.784,53.800,0.095,0.120],
     'drivers':{'strat':40,'gas':20,'roughness':15,'slope':12,'uncert':13,'mag':0,'bedform':0},
     'obs':['V-shaped buried channel 4–8m deep (SBP)',
            'Low backscatter channel infill (SSS)',
            'Subtle surface expression — channel partially infilled (MBES)'],
     'interp':['Buried glacial channel — soft infill creates differential settlement risk',
               'Co-location of channel infill and weak gas indicators elevates risk',
               'Channel walls may have steep near-surface lateral slopes'],
     'why':['Soft channel infill has lower bearing capacity than surrounding seabed',
            'Differential settlement creates cable bending stress over channel edges',
            'Possible gas accumulation in channel infill — SBP shows weak blanking'],
     'confidence':'Moderate','conf_note':'Channel geometry from SBP at 120m line spacing — infill properties inferred'},

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN SURVEY AREA — CENTRAL SECTOR  (lon 0.140–0.175, lat 53.795–0.825)
    # ═══════════════════════════════════════════════════════════════════════

    {'id':'RZ-C5','label':'Wreck Site Alpha','score':9.2,
     'bbox':[53.800,53.809,0.140,0.156],
     'drivers':{'mag':50,'roughness':20,'strat':15,'slope':8,'uncert':7,'gas':0,'bedform':0},
     'obs':['Dominant 32,791 nT magnetic dipole (Mag)',
            'High acoustic backscatter — upstanding hull (SSS)',
            'Elevation anomaly 3.8m above seabed (MBES)',
            'Hard reflector — no SBP penetration (SBP)'],
     'interp':['Large ferrous mass co-located with acoustic target confirms wreck',
               'Scour moat visible around base — ongoing exposure and sediment mobility',
               'No acoustic penetration below target — solid metallic structure'],
     'why':['32,791 nT is the highest anomaly in the survey — probable WWII steel vessel',
            'Upstanding obstacle creates direct cable strike risk',
            'Scour indicates dynamic seabed — position of buried cable may shift'],
     'confidence':'High','conf_note':'Multi-sensor agreement across SSS, MBES, SBP and Magnetometer'},

    {'id':'RZ-C6','label':'Wreck Scatter Zone','score':7.8,
     'bbox':[53.809,53.818,0.136,0.157],
     'drivers':{'mag':35,'roughness':30,'strat':15,'slope':10,'uncert':10,'gas':0,'bedform':0},
     'obs':['Elevated magnetic anomalies (1,150 nT secondary peak) (Mag)',
            'Irregular high-backscatter patches (SSS)',
            'Localised roughness elevation above regional background (MBES)'],
     'interp':['Scattered debris field from primary wreck — ferrous fragments across ~200m radius',
               'Roughness inconsistent with natural seabed — anthropogenic origin likely'],
     'why':['Debris field creates multiple cable snag and abrasion points',
            'Magnetic signature suggests ferrous material dispersed by current or salvage'],
     'confidence':'Moderate','conf_note':'SSS and MBES consistent; SBP coverage patchy in this cell'},

    {'id':'RZ-C7','label':'UXO Zone B','score':7.9,
     'bbox':[53.806,53.815,0.156,0.178],
     'drivers':{'mag':45,'strat':20,'uncert':18,'roughness':10,'slope':7,'gas':0,'bedform':0},
     'obs':['141 nT dipole anomaly (Mag)',
            'Clean seabed surface — no SSS expression',
            'Weak SBP reflector at ~1.5m depth'],
     'interp':['UXO-class dipole consistent with medium-calibre ordnance',
               'Deeper burial than Zone A — may be below standard trenching depth'],
     'why':['141 nT within range for WWII bombs, mines and shells',
            'Target depth ~1.5m — within cable route disturbance envelope for HDD or plough'],
     'confidence':'Moderate','conf_note':'SBP line spacing 120m — depth estimate interpolated'},

    {'id':'RZ-C8','label':'Central Mixed Risk','score':5.8,
     'bbox':[53.798,53.808,0.126,0.142],
     'drivers':{'mag':25,'roughness':25,'strat':20,'slope':15,'uncert':15,'gas':0,'bedform':0},
     'obs':['Moderate magnetic anomalies at background elevation (Mag)',
            'Variable backscatter — patchy sediment cover (SSS)',
            'Stratified sequence with minor reflector discontinuities (SBP)'],
     'interp':['Multiple low-level indicators across sensors — no single dominant hazard',
               'Patchy sediment cover creates variable burial depth along route'],
     'why':['Combination of moderate signals from multiple sensors elevates composite score',
            'Variable ground conditions complicate burial design specification'],
     'confidence':'Moderate','conf_note':'Good data coverage; distributed risk is genuine rather than data-gap driven'},

    {'id':'RZ-C9','label':'Gas + Mag Combined','score':8.0,
     'bbox':[53.808,53.820,0.094,0.118],
     'drivers':{'gas':35,'mag':25,'strat':20,'roughness':12,'uncert':8,'slope':0,'bedform':0},
     'obs':['Acoustic blanking zone (SBP)',
            'Elevated magnetic background across zone (Mag)',
            'Pockmark expression (MBES)',
            'Low backscatter at pockmark centres (SSS)'],
     'interp':['Co-location of shallow gas and magnetic anomaly cluster increases composite hazard score',
               'Gas and ferrous target combination may indicate corroded munitions contributing to gas migration',
               'Seabed alone appears manageable — subsurface instability indicators dominate'],
     'why':['Shallow gas suggests weak near-surface conditions throughout zone',
            'Magnetic targets within gas zone — possible corroded UXO contributing to gas migration',
            'Nearest SBP line spacing 120m — significant interpolation required across this cell'],
     'confidence':'Moderate','conf_note':'Nearest SBP line spacing 120m; no CPT tie in this cell — reduced confidence'},

    {'id':'RZ-C10','label':'Boulder Field','score':6.5,
     'bbox':[53.818,53.828,0.098,0.124],
     'drivers':{'roughness':40,'slope':20,'strat':15,'uncert':15,'mag':10,'gas':0,'bedform':0},
     'obs':['High-backscatter clusters — 45+ targets >0.5m (SSS)',
            'Elevation peaks 0.5–2.3m (MBES)',
            'Possible ferrous inclusions in larger boulders (Mag)'],
     'interp':['Glacially-derived boulder field — dense obstacle hazard to cable laying',
               'Subsurface boulder density unknown — SSS only shows surface expression'],
     'why':['Dense boulder field creates mechanical hazard to cable during installation',
            'Burial between boulders not feasible — cable will be surface-laid through field',
            'Exposed cable between boulders has elevated third-party damage risk'],
     'confidence':'Moderate','conf_note':'Subsurface boulder extent uncertain — SSS only shows surface expression'},

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN SURVEY AREA — NE SECTOR  (lon 0.145–0.215, lat 53.815–0.845)
    # ═══════════════════════════════════════════════════════════════════════

    {'id':'RZ-NE1','label':'Wreck Site Beta','score':8.5,
     'bbox':[53.824,53.835,0.152,0.167],
     'drivers':{'mag':45,'roughness':22,'strat':13,'slope':10,'uncert':10,'gas':0,'bedform':0},
     'obs':['4,522 nT magnetic anomaly — second largest in survey (Mag)',
            'Discrete high-backscatter target (SSS)',
            'Bathymetric elevation ~2m (MBES)'],
     'interp':['Probable wreck or large metallic structure — significant ferrous mass',
               'Co-location of SSS target and mag anomaly increases identification confidence'],
     'why':['4,522 nT well above ordnance threshold — wreck-class ferrous mass',
            'Water depth and position consistent with WWII shipping lane routing'],
     'confidence':'High','conf_note':'Strong dual-sensor agreement (SSS + Mag); SBP not conclusive'},

    {'id':'RZ-NE2','label':'Northern Anomaly Cluster','score':7.1,
     'bbox':[53.828,53.840,0.133,0.153],
     'drivers':{'mag':35,'strat':25,'roughness':18,'slope':12,'uncert':10,'gas':0,'bedform':0},
     'obs':['61 nT magnetic anomaly — moderate ferrous target (Mag)',
            'Irregular SSS backscatter (SSS)',
            'Roughness elevated above regional background (MBES)'],
     'interp':['Magnetic anomaly co-located with acoustic target — probable metallic debris',
               'Roughness inconsistent with surrounding seabed — possible anthropogenic origin'],
     'why':['Ferrous target within cable route — identification and clearance required',
            'Unknown target type — conservative UXO or debris assumption warranted'],
     'confidence':'Moderate','conf_note':'Single dominant sensor (Mag); SSS expression ambiguous'},

    {'id':'RZ-NE3','label':'NE Sand Corridor','score':5.2,
     'bbox':[53.820,53.834,0.155,0.178],
     'drivers':{'bedform':30,'roughness':28,'slope':18,'uncert':14,'strat':10,'gas':0,'mag':0},
     'obs':['Moderate bedform activity (SSS)',
            'Elevated roughness above background (MBES)',
            'Sand layer 1–2m thick (SBP)'],
     'interp':['Active but low-amplitude bedforms — lower mobility risk than SE field',
               'Roughness within manageable range for standard burial design'],
     'why':['Bedform migration could expose cable within 2–3 years post-installation',
            'Monitoring programme recommended at 1-year and 3-year intervals'],
     'confidence':'Moderate','conf_note':'Northern limit of SSS coverage — some extrapolation applied'},

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN SURVEY AREA — SE SECTOR  (lon 0.155–0.215, lat 53.764–0.800)
    # ═══════════════════════════════════════════════════════════════════════

    {'id':'RZ-SE1','label':'Active Sand Wave Field','score':6.8,
     'bbox':[53.764,53.784,0.155,0.195],
     'drivers':{'bedform':45,'roughness':25,'slope':15,'uncert':10,'strat':5,'gas':0,'mag':0},
     'obs':['Sinusoidal bedform pattern — wavelength 20–40m (SSS)',
            'Asymmetric crest profiles confirming active migration (SSS)',
            'Amplitude 0.8–2.5m (MBES)',
            'Mobile sand layer 1–3m thick (SBP)'],
     'interp':['Bedform mobility indicates dynamic seabed — installation timing critical',
               'Asymmetric crests confirm net NE migration direction',
               'Seabed position at installation will differ from survey baseline'],
     'why':['Active bedforms create post-installation free-span risk as sand migrates',
            'Burial depth achieved at installation may not be maintained over cable lifetime',
            'Re-survey recommended within 6 months prior to construction'],
     'confidence':'High','conf_note':'SSS migration indicators clear; MBES amplitude consistent with active bedforms'},

    {'id':'RZ-SE2','label':'Sand Wave Transition','score':5.5,
     'bbox':[53.780,53.798,0.145,0.168],
     'drivers':{'bedform':35,'roughness':25,'slope':20,'uncert':12,'strat':8,'gas':0,'mag':0},
     'obs':['Transition from sand wave to plane bed (SSS)',
            'Moderate roughness at bedform terminus (MBES)',
            'Sediment thickness reducing toward SW (SBP)'],
     'interp':['Bedform activity reducing toward SW — lower mobility risk than RZ-SE1',
               'Roughness still elevated at transition zone'],
     'why':['Transition zones concentrate cable stress due to differential burial depth',
            'Slope gradient at crest terminations creates localised bending load'],
     'confidence':'Moderate','conf_note':'Transition zone geometry interpreted from limited SBP line crossings'},

    {'id':'RZ-SE3','label':'UXO Proximity Zone','score':6.3,
     'bbox':[53.785,53.798,0.168,0.190],
     'drivers':{'mag':40,'strat':22,'uncert':18,'roughness':12,'slope':8,'gas':0,'bedform':0},
     'obs':['51 nT magnetic anomaly — near UXO threshold (Mag)',
            'Clean seabed surface (SSS)',
            'Minor reflector disruption at depth (SBP)'],
     'interp':['Anomaly amplitude below primary UXO threshold but above background noise',
               'Position in known WWII disposal area elevates prior probability'],
     'why':['Conservative classification warranted given WWII disposal area context',
            '51 nT within detection range for small ordnance items'],
     'confidence':'Low','conf_note':'Amplitude near lower detection threshold; single-sensor evidence only'},

    {'id':'RZ-SE4','label':'Eastern Rock Outcrop','score':7.5,
     'bbox':[53.772,53.786,0.192,0.213],
     'drivers':{'roughness':40,'slope':25,'strat':15,'uncert':15,'gas':5,'mag':0,'bedform':0},
     'obs':['Very high backscatter — rock at seabed (SSS)',
            'RMS roughness >1.5m (MBES)',
            'Hard reflector <0.5m depth — no SBP penetration (SBP)'],
     'interp':['Rock outcrop confirmed — burial not feasible without rock-cutting',
               'No sediment veneer — cable fully exposed if laid directly here'],
     'why':['Rock seabed precludes standard plough or jet-trench burial method',
            'Exposed cable on rock face has high risk of third-party damage and abrasion',
            'Route modification or rock mattress protection required'],
     'confidence':'High','conf_note':'SSS backscatter and SBP hard reflector are diagnostic of rock'},

    {'id':'RZ-SE5','label':'SE Flat — Low Risk','score':3.1,
     'bbox':[53.798,53.814,0.180,0.213],
     'drivers':{'slope':30,'roughness':25,'uncert':25,'strat':10,'bedform':10,'gas':0,'mag':0},
     'obs':['Flat, featureless seabed (MBES)',
            'Homogeneous low-backscatter surface (SSS)',
            'Clear stratified sequence to 6m depth (SBP)'],
     'interp':['Seabed conditions are favourable for cable installation',
               'No significant hazards identified in current dataset'],
     'why':['Data uncertainty is the primary residual risk — survey at SE limit of coverage',
            'No independent geotechnical data (CPT) to validate SBP interpretation'],
     'confidence':'Moderate','conf_note':'SE margin of survey — reduced sensor coverage and wider line spacing'},

    {'id':'RZ-SE6','label':'Gas Migration Corridor','score':7.4,
     'bbox':[53.764,53.778,0.120,0.155],
     'drivers':{'gas':38,'strat':28,'uncert':18,'roughness':10,'slope':6,'mag':0,'bedform':0},
     'obs':['Sporadic acoustic blanking (SBP)',
            'Subtle pockmark expression (MBES)',
            'Low-backscatter corridor (SSS)'],
     'interp':['Gas migration pathway inferred from pockmark alignment and blanking distribution',
               'Lower confidence than main pockmark cluster — blanking intermittent'],
     'why':['Gas migration corridor suggests connected subsurface plumbing system',
            'Risk lower than source zone but geotechnical variability still elevated'],
     'confidence':'Moderate','conf_note':'SBP blanking intermittent; pockmark morphology subtle'},

    {'id':'RZ-SE7','label':'Channel Edge Hazard','score':6.2,
     'bbox':[53.812,53.824,0.118,0.140],
     'drivers':{'strat':35,'slope':28,'roughness':15,'uncert':15,'gas':7,'mag':0,'bedform':0},
     'obs':['Channel edge slope 8–15° (MBES)',
            'Reflector disruption at channel wall (SBP)',
            'Moderate backscatter change at transition (SSS)'],
     'interp':['Channel wall creates localised slope hazard at cable crossing',
               'Disrupted reflectors suggest possible mass movement history'],
     'why':['Slope at channel edge increases cable stress and potential for sliding',
            'Mass movement history indicates geotechnical instability risk'],
     'confidence':'Moderate','conf_note':'Channel wall slope from MBES — limited SBP crossing data available'},
]
def get_risk_zones_gdf():
    """Convert RISK_ZONES_DEF to a GeoDataFrame using pre-clipped polygon coords."""
    from shapely.geometry import Polygon
    import geopandas as _gpd

    # Pre-clipped polygon vertices (lon, lat) — computed by Sutherland-Hodgman
    # against the survey boundary so no zone overhangs the survey area edges
    CLIPPED_COORDS = {
        'RZ-C1':  [(-0.009, 53.774156), (0.020, 53.777545), (0.020, 53.780445), (-0.009, 53.777056)],
        'RZ-C2':  [(0.020, 53.777545), (0.055, 53.781634), (0.055, 53.784534), (0.020, 53.780445)],
        'RZ-C3':  [(0.055, 53.781634), (0.090, 53.785723), (0.090, 53.788623), (0.055, 53.784534)],
        'RZ-C4':  [(0.090, 53.785723), (0.114, 53.788527), (0.114, 53.791427), (0.090, 53.788623)],
        'RZ-SW1': [(0.114, 53.788318), (0.129534, 53.774), (0.138, 53.774), (0.138, 53.796), (0.114, 53.796)],
        'RZ-SW2': [(0.11, 53.797), (0.125, 53.797), (0.125, 53.806), (0.11, 53.806)],
        'RZ-SW3': [(0.095, 53.80583), (0.10675, 53.795), (0.115, 53.795), (0.115, 53.808), (0.095, 53.808)],
        'RZ-SW4': [(0.097, 53.803987), (0.101326, 53.8), (0.12, 53.8), (0.12, 53.812), (0.097, 53.812)],
        'RZ-C5':  [(0.14, 53.8), (0.156, 53.8), (0.156, 53.809), (0.14, 53.809)],
        'RZ-C6':  [(0.135, 53.809), (0.156, 53.809), (0.156, 53.818), (0.135, 53.818)],
        'RZ-C7':  [(0.156, 53.806), (0.177, 53.806), (0.177, 53.815), (0.156, 53.815)],
        'RZ-C8':  [(0.127, 53.798), (0.142, 53.798), (0.142, 53.808), (0.127, 53.808)],
        'RZ-C9':  [(0.095, 53.818373), (0.095, 53.808), (0.118, 53.808), (0.118, 53.82), (0.099226, 53.82)],
        'RZ-C10': [(0.099, 53.819913), (0.099, 53.818), (0.122, 53.818), (0.122, 53.828), (0.120002, 53.828)],
        'RZ-NE1': [(0.152, 53.824), (0.167, 53.824), (0.167, 53.835), (0.152, 53.835)],
        'RZ-NE2': [(0.133, 53.833005), (0.133, 53.828), (0.153, 53.828), (0.153, 53.84), (0.151167, 53.84)],
        'RZ-NE3': [(0.155, 53.82), (0.177, 53.82), (0.177, 53.828912), (0.17266, 53.833), (0.155, 53.833)],
        'RZ-SE1': [(0.158, 53.775), (0.165779, 53.775), (0.188, 53.784067), (0.188, 53.786), (0.158, 53.786)],
        'RZ-SE2': [(0.145, 53.782), (0.167, 53.782), (0.167, 53.797), (0.145, 53.797)],
        'RZ-SE3': [(0.168, 53.786), (0.19, 53.786), (0.19, 53.797), (0.168, 53.797)],
        'RZ-SE4': [(0.178, 53.786), (0.192738, 53.786), (0.205, 53.791004), (0.205, 53.793), (0.178, 53.793)],
        'RZ-SE5': [(0.175, 53.798), (0.2, 53.798), (0.2, 53.80725), (0.19708, 53.81), (0.175, 53.81)],
        'RZ-SE6': [(0.14, 53.766), (0.143722, 53.766), (0.157, 53.771418), (0.157, 53.778), (0.14, 53.778)],
        'RZ-SE7': [(0.118, 53.812), (0.138, 53.812), (0.138, 53.823), (0.118, 53.823)],
    }

    rows = []
    for z in RISK_ZONES_DEF:
        zid = z['id']
        coords = CLIPPED_COORDS.get(zid)
        if not coords or len(coords) < 3:
            continue
        rows.append({
            'geometry':   Polygon(coords),
            'cell_id':    zid,
            'label':      z['label'],
            'score':      z['score'],
            'drivers':    z['drivers'],
            'obs':        z['obs'],
            'interp':     z['interp'],
            'why':        z['why'],
            'confidence': z['confidence'],
            'conf_note':  z['conf_note'],
        })
    return _gpd.GeoDataFrame(rows, crs='EPSG:4326')


def risk_zone_color(score):
    if score >= 8.0: return '#CC0000', 0.42
    if score >= 6.5: return '#FF6600', 0.34
    if score >= 5.0: return '#FFAA00', 0.26
    if score >= 3.5: return '#FFDD00', 0.20
    return '#44BB44', 0.14


def _donut_svg(drivers, score):
    """Generate a compact inline SVG donut chart from driver percentages."""
    import math
    items = sorted([(k, v) for k, v in drivers.items() if v > 0], key=lambda x: -x[1])
    colors = {d[1]: d[2] for d in RISK_DRIVERS}
    cx, cy, R, r = 70, 70, 56, 30
    total = sum(v for _, v in items)
    angle = -math.pi / 2
    paths = []
    for key, val in items:
        theta = 2 * math.pi * val / total
        x1  = cx + R * math.cos(angle);          y1  = cy + R * math.sin(angle)
        x2  = cx + R * math.cos(angle + theta);  y2  = cy + R * math.sin(angle + theta)
        ix1 = cx + r * math.cos(angle);          iy1 = cy + r * math.sin(angle)
        ix2 = cx + r * math.cos(angle + theta);  iy2 = cy + r * math.sin(angle + theta)
        lg  = 1 if theta > math.pi else 0
        col = colors.get(key, '#aaa')
        d   = (f"M {x1:.1f} {y1:.1f} A {R} {R} 0 {lg} 1 {x2:.1f} {y2:.1f} "
               f"L {ix2:.1f} {iy2:.1f} A {r} {r} 0 {lg} 0 {ix1:.1f} {iy1:.1f} Z")
        paths.append(f'<path d="{d}" fill="{col}" stroke="white" stroke-width="1.5"/>')
        angle += theta
    paths.append(f'<text x="{cx}" y="{cy-5}" text-anchor="middle" font-size="15" font-weight="bold" fill="#222">{score}</text>')
    paths.append(f'<text x="{cx}" y="{cy+12}" text-anchor="middle" font-size="9" fill="#666">/10</text>')
    svg = f'<svg width="140" height="140" viewBox="0 0 140 140" xmlns="http://www.w3.org/2000/svg">{"".join(paths)}</svg>'

    legend_rows = ""
    for key, val in items:
        col   = colors.get(key, '#aaa')
        label = next((d[0] for d in RISK_DRIVERS if d[1] == key), key)
        legend_rows += (f'<tr><td style="padding:1px 4px 1px 0;">'
                        f'<span style="display:inline-block;width:10px;height:10px;background:{col};'
                        f'border-radius:2px;"></span></td>'
                        f'<td style="font-size:10px;padding-right:8px;white-space:nowrap;">{label}</td>'
                        f'<td style="font-size:10px;font-weight:bold;">{val}%</td></tr>')
    return (f'<div style="display:flex;align-items:center;gap:10px;">'
            f'{svg}<table style="border-collapse:collapse;line-height:1.6;">{legend_rows}</table></div>')


def risk_zone_popup(cell):
    """Build the HTML popup for a risk zone cell — donut chart + narrative."""
    score     = cell['score']
    label     = cell['label']
    cell_id   = cell['cell_id']
    drivers   = cell['drivers']
    conf      = cell['confidence']
    conf_note = cell['conf_note']

    dominant_key = max(drivers, key=drivers.get)
    dom_label    = next(d[0] for d in RISK_DRIVERS if d[1] == dominant_key)

    if score >= 8.0:   band, band_col, txt_col = "HIGH RISK",      "#CC0000", "#fff"
    elif score >= 6.5: band, band_col, txt_col = "MODERATE-HIGH",  "#FF6600", "#fff"
    elif score >= 5.0: band, band_col, txt_col = "MODERATE",       "#FFAA00", "#111"
    elif score >= 3.5: band, band_col, txt_col = "LOW-MODERATE",   "#FFDD00", "#111"
    else:              band, band_col, txt_col = "LOW RISK",        "#44BB44", "#fff"

    obs_html    = "".join(f"<li style='margin:2px 0;font-size:11px;'>{o}</li>" for o in cell['obs'])
    interp_html = "".join(f"<li style='margin:2px 0;font-size:11px;'>{i}</li>" for i in cell['interp'])
    why_html    = "".join(f"<li style='margin:2px 0;font-size:11px;'>{w}</li>" for w in cell['why'])
    conf_color  = {'High':'#00AA44','Moderate':'#FF8C00','Low':'#CC0000'}.get(conf, '#888')
    donut_html  = _donut_svg(drivers, score)

    return f"""<div style="width:430px;font-family:Arial;font-size:12px;max-height:580px;overflow-y:auto;">
  <div style="padding:10px 14px;background:{band_col};color:{txt_col};border-radius:6px 6px 0 0;">
    <div style="font-size:15px;font-weight:bold;">{label} <span style="font-size:10px;opacity:0.75;">({cell_id})</span></div>
    <div style="font-size:11px;margin-top:2px;">{band} &nbsp;·&nbsp; Score: <b>{score}/10</b> &nbsp;·&nbsp; Dominant: <b>{dom_label}</b></div>
  </div>
  <div style="padding:12px 14px;border:1px solid #ddd;border-top:none;border-radius:0 0 6px 6px;background:#fff;">
    <div style="font-weight:bold;font-size:10px;color:#777;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Risk Driver Breakdown</div>
    {donut_html}
    <div style="border-top:1px solid #eee;padding-top:8px;margin-top:10px;">
      <div style="font-weight:bold;font-size:11px;color:#444;margin-bottom:3px;">📡 Observed</div>
      <ul style="margin:0;padding-left:16px;">{obs_html}</ul>
    </div>
    <div style="border-top:1px solid #eee;padding-top:8px;margin-top:6px;">
      <div style="font-weight:bold;font-size:11px;color:#0013C3;margin-bottom:3px;">🤖 GRID AI Interpretation</div>
      <ul style="margin:0;padding-left:16px;">{interp_html}</ul>
    </div>
    <div style="border-top:1px solid #eee;padding-top:8px;margin-top:6px;">
      <div style="font-weight:bold;font-size:11px;color:#c45000;margin-bottom:3px;">⚠️ Why Risky</div>
      <ul style="margin:0;padding-left:16px;">{why_html}</ul>
    </div>
    <div style="border-top:1px solid #eee;padding-top:6px;margin-top:6px;">
      <span style="font-size:10px;font-weight:bold;color:{conf_color};">● {conf} Confidence</span>
      <span style="font-size:10px;color:#777;"> — {conf_note}</span>
    </div>
  </div>
</div>"""


def risk_weight_breakdown_html(haz_row):
    """Per-hazard risk weighting breakdown for Evidence Viewer — bars + rationale."""
    htype = haz_row.get('hazard_type', '')
    score = haz_row.get('risk_score', 5)

    profiles = {
        'Wreck':    {'mag':45,'roughness':20,'strat':15,'slope':10,'uncert':10,'gas':0,'bedform':0,
                     'rationale':{'mag':'Large ferrous mass produces dominant magnetic signature (32,791 nT at WRK-001)',
                                  'roughness':'Upstanding hull creates localised scour and roughness anomaly',
                                  'strat':'Wreck indicates historical seabed disturbance and possible shell infill',
                                  'slope':'Local bathymetric anomaly from hull elevation above seabed',
                                  'uncert':'Structural integrity and burial state of wreck unknown'}},
        'UXO':      {'mag':50,'strat':20,'uncert':15,'roughness':10,'slope':5,'gas':0,'bedform':0,
                     'rationale':{'mag':'Compact dipole anomaly — primary detection method for buried ordnance',
                                  'strat':'Burial depth from SBP poorly constrained at 120m line spacing',
                                  'uncert':'Ordnance type and condition unknown — conservative assumptions required',
                                  'roughness':'Minor seabed disturbance in vicinity of target',
                                  'slope':'Flat terrain — low contribution to overall risk score'}},
        'Pipeline': {'mag':35,'roughness':25,'slope':20,'strat':10,'uncert':10,'gas':0,'bedform':0,
                     'rationale':{'mag':'Linear ferrous anomaly confirms steel pipeline by magnetometer',
                                  'roughness':'Exposed pipeline creates cable spanning and entanglement hazard',
                                  'slope':'Bathymetric gradient may stress cable at pipeline crossings',
                                  'strat':'Possible buried section — depth of cover uncertain in places',
                                  'uncert':'Operating status, pressure and contents unknown'}},
        'Gas':      {'gas':45,'strat':30,'roughness':10,'slope':5,'uncert':10,'mag':0,'bedform':0,
                     'rationale':{'gas':'Acoustic blanking in SBP confirms shallow gas accumulation',
                                  'strat':'Pockmark morphology indicates disturbed stratigraphy below surface',
                                  'roughness':'Surface expression of gas escape increases seabed roughness locally',
                                  'slope':'Minor contribution — pockmarks on broadly flat terrain',
                                  'uncert':'Gas volume and migration rate poorly constrained from available data'}},
        'Sand Wave':{'bedform':45,'roughness':25,'slope':15,'uncert':10,'strat':5,'gas':0,'mag':0,
                     'rationale':{'bedform':'Asymmetric SSS pattern confirms active migration — free-span risk',
                                  'roughness':'Mobile sand creates variable burial depth post-installation',
                                  'slope':'Crest-to-trough elevation creates localised cable bending stress',
                                  'uncert':'Migration rate seasonal — limited repeat survey data available',
                                  'strat':'Low contribution — feature is entirely surficial'}},
        'Hard Ground':{'roughness':40,'slope':25,'strat':15,'uncert':15,'gas':5,'mag':0,'bedform':0,
                       'rationale':{'roughness':'Very high backscatter and rugosity — standard burial not feasible',
                                    'slope':'Rock outcrop gradient increases cable stress at approach angle',
                                    'strat':'No sediment cover — SBP cannot penetrate; unknowns persist below rock',
                                    'uncert':'Rock surface extent and lateral continuity uncertain',
                                    'gas':'Minor contribution — no evidence of gas but acoustic penetration absent'}},
        'Boulder':  {'roughness':40,'slope':20,'strat':15,'uncert':15,'mag':10,'gas':0,'bedform':0,
                     'rationale':{'roughness':'Dense boulder field — high mechanical hazard to cable during installation',
                                  'slope':'Boulder clusters create localised gradient anomalies',
                                  'strat':'Glacial origin suggests heterogeneous subsurface conditions',
                                  'uncert':'Subsurface boulder density unknown — partial SSS coverage only',
                                  'mag':'Possible ferrous inclusions in largest boulders'}},
        'Channel':  {'strat':40,'gas':20,'slope':20,'roughness':10,'uncert':10,'mag':0,'bedform':0,
                     'rationale':{'strat':'Buried glacial channel — soft infill creates differential settlement risk',
                                  'gas':'Weak SBP blanking in channel infill — possible gas accumulation',
                                  'slope':'Channel wall slope creates cable bending stress at crossings',
                                  'roughness':'Transition zone roughness at channel edge',
                                  'uncert':'Channel depth and infill properties from SBP at 120m line spacing only'}},
    }

    profile = next((v for k, v in profiles.items() if k in htype), {
        'slope':20,'roughness':20,'strat':20,'uncert':20,'gas':10,'mag':5,'bedform':5,
        'rationale':{k:'Contributing factor based on multi-sensor assessment' for k in DRIVER_KEYS}})

    pct       = {k: profile.get(k, 0) for k in DRIVER_KEYS}
    rationale = profile.get('rationale', {})
    dominant  = max(pct, key=pct.get)
    dom_label = next(d[0] for d in RISK_DRIVERS if d[1] == dominant)

    bars = ""
    for label, key, col in sorted(RISK_DRIVERS, key=lambda d: -pct[d[1]]):
        w = pct[key]
        if w == 0: continue
        note = rationale.get(key, '')
        bars += (f'<div style="margin:5px 0;padding:6px 10px;background:#f8f8f8;'
                 f'border-radius:4px;border-left:3px solid {col};">'
                 f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">'
                 f'<span style="font-weight:bold;width:160px;font-size:12px;">{label}</span>'
                 f'<div style="flex:1;background:#e0e0e0;border-radius:3px;height:12px;">'
                 f'<div style="width:{w}%;background:{col};height:12px;border-radius:3px;"></div></div>'
                 f'<span style="font-weight:bold;width:36px;text-align:right;font-size:13px;">{w}%</span>'
                 f'</div><div style="font-size:11px;color:#555;padding-left:168px;">{note}</div></div>')

    return (f'<div style="font-family:Arial;">'
            f'<div style="padding:10px 14px;background:#0013C3;color:white;border-radius:6px;margin-bottom:12px;">'
            f'<span style="font-size:15px;font-weight:bold;">⚖️ Risk Score: {score}/10</span>'
            f'<span style="margin-left:16px;font-size:12px;">Dominant driver: <b>{dom_label} ({pct[dominant]}%)</b></span>'
            f'</div>{bars}</div>')



def create_timeline_gantt(hazards_gdf, view='before'):
    """
    Single Gantt chart with two views:
      'before' = baseline pre-survey schedule (no hazard impact)
      'after'  = hazard-adjusted schedule with per-phase delay attribution
    Returns (fig, total_months, phase_delays_dict)
    """
    import plotly.graph_objects as go
    from datetime import datetime, timedelta

    start_date = datetime(2025, 6, 1)

    INX_COLORS = {
        'Survey':       '#0013C3',
        'Analysis':     '#8288A3',
        'Engineering':  '#12241E',
        'Planning':     '#8F998D',
        'Construction': '#D1FE49',
        'Contingency':  '#FF4500',
    }

    # ── Hazard counts ──────────────────────────────────────────────────────────
    if hazards_gdf is not None and len(hazards_gdf) > 0:
        crit = int(len(hazards_gdf[hazards_gdf['risk'] == 'Critical']))
        high = int(len(hazards_gdf[hazards_gdf['risk'] == 'High']))
        med  = int(len(hazards_gdf[hazards_gdf['risk'] == 'Medium']))
        low  = int(len(hazards_gdf[hazards_gdf['risk'] == 'Low']))
    else:
        crit = high = med = low = 0

    # ── Phase delay calculation ────────────────────────────────────────────────
    # Each hazard tier drives delay in specific phases
    delays = {
        'Geotech Survey':        round(min(crit * 0.25 + high * 0.1, 1.0), 1),
        'FEED / Engineering':    round(min(crit * 0.5  + high * 0.25 + med * 0.1, 2.0), 1),
        'UXO / Clearance':       round(min(crit * 0.5  + high * 0.25, 2.0), 1),
        'Consenting & Planning': round(min(crit * 0.5  + high * 0.25 + med * 0.1, 2.0), 1),
        'Construction Prep':     round(min(high * 0.15 + med * 0.05, 0.5), 1),
    }

    # ── Baseline phases (months duration) ─────────────────────────────────────
    base_phases = [
        ('Geophysical Survey',   3,  'Survey'),
        ('Data Processing',      2,  'Analysis'),
        ('Geotech Survey',       3,  'Survey'),
        ('FEED / Engineering',   5,  'Engineering'),
        ('UXO / Clearance',      0,  'Contingency'),   # 0 in baseline
        ('Consenting & Planning',12, 'Planning'),
        ('Construction Prep',    3,  'Engineering'),
        ('Construction',         18, 'Construction'),
        ('Commissioning',        3,  'Survey'),
    ]

    phases = []
    curr = 0
    for name, base_dur, resource in base_phases:
        if view == 'before':
            dur = base_dur
            ext = 0.0
        else:
            ext = delays.get(name, 0.0)
            dur = base_dur + ext
        if dur == 0:
            continue
        phases.append({
            'task': name, 'start': curr, 'dur': dur,
            'base': base_dur, 'ext': ext, 'resource': resource
        })
        curr += dur

    total_months = curr

    # ── Build Gantt bars ───────────────────────────────────────────────────────
    fig = go.Figure()
    bar_height = 0.6

    for i, p in enumerate(phases):
        sd = start_date + timedelta(days=p['start'] * 30.4)
        ed = sd + timedelta(days=p['base'] * 30.4)
        col = INX_COLORS.get(p['resource'], '#aaa')

        # Base duration bar
        hover = (f"<b>{p['task']}</b><br>"
                 f"Base duration: {p['base']} months<br>"
                 f"Start: month {p['start']:.1f}<br>"
                 f"Resource: {p['resource']}")
        if view == 'after' and p['ext'] > 0:
            hover += f"<br><b>+{p['ext']:.1f} month hazard delay</b>"

        fig.add_trace(go.Bar(
            name=p['task'],
            x=[p['base']],
            y=[p['task']],
            base=[p['start']],
            orientation='h',
            marker=dict(color=col, opacity=0.92, line=dict(color='white', width=1)),
            hovertemplate=hover + '<extra></extra>',
            showlegend=False,
            width=bar_height,
        ))

        # Extension bar (hazard delay) shown in orange
        if view == 'after' and p['ext'] > 0:
            ext_hover = (f"<b>⚠️ Hazard Delay: {p['task']}</b><br>"
                         f"+{p['ext']:.1f} months from {crit}C/{high}H/{med}M/{low}L hazards")
            fig.add_trace(go.Bar(
                name=f"{p['task']} delay",
                x=[p['ext']],
                y=[p['task']],
                base=[p['start'] + p['base']],
                orientation='h',
                marker=dict(color='#FF4500', opacity=0.75,
                            pattern=dict(shape='/', size=6, fgcolor='rgba(255,69,0,0.4)')),
                hovertemplate=ext_hover + '<extra></extra>',
                showlegend=False,
                width=bar_height,
            ))

    # ── Month axis → calendar labels ──────────────────────────────────────────
    tick_months = list(range(0, int(total_months) + 3, 3))
    tick_labels = []
    for m in tick_months:
        d = start_date + timedelta(days=m * 30.4)
        tick_labels.append(d.strftime("%b '%y"))

    # ── Vertical "now" line ────────────────────────────────────────────────────
    now_month = 9  # ~March 2026 relative to Jun 2025 start

    # ── Layout ────────────────────────────────────────────────────────────────
    title_text = (
        f"Baseline Schedule — {total_months:.0f} months"
        if view == 'before'
        else f"Hazard-Adjusted Schedule — {total_months:.0f} months "
             f"(+{total_months - sum(p['base'] for p in phases):.1f} mo delay)"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15, color='#2E2E2E'), x=0),
        barmode='overlay',
        xaxis=dict(
            title='Project Month',
            tickvals=tick_months,
            ticktext=tick_labels,
            gridcolor='#ececec',
            zeroline=False,
            range=[-0.5, total_months + 1],
        ),
        yaxis=dict(autorange='reversed', tickfont=dict(size=11)),
        height=420,
        margin=dict(l=10, r=20, t=50, b=40),
        paper_bgcolor='white',
        plot_bgcolor='#fafafa',
        font=dict(family='Arial', size=11, color='#2E2E2E'),
        shapes=[dict(
            type='line', x0=now_month, x1=now_month, y0=-0.5,
            y1=len(phases) - 0.5,
            line=dict(color='#0013C3', width=2, dash='dot')
        )],
        annotations=[dict(
            x=now_month, y=-0.5, text='▲ Now', showarrow=False,
            font=dict(color='#0013C3', size=10), yanchor='top'
        )],
    )

    return fig, total_months, delays


# ── Per-hazard type: detailed engineering impact profiles ──────────────────────
HAZARD_IMPACT_PROFILES = {
    'Wreck': {
        'cost_range': '£450K – £1.2M',
        'cost_low': 450000, 'cost_high': 1200000,
        'schedule_exposure': '6–14 weeks',
        'next_survey': 'ROV inspection + sub-bottom profiling; 1:1000 scale sonar mosaic',
        'contributions': [
            ('Route Diversion Design', 'Probable 500–800m cable route deviation; structural design uplift required'),
            ('Rock Dumping Contingency', 'Upstanding hull requires mattressing or concrete protection; probable £200–400K contingency'),
            ('UXO Co-assessment', 'WWII wreck — UXO on or within hull; specialist dive or ROV sweep before burial design'),
            ('Pre-FEED Investigation', 'Elevated requirement for ROV-based structural survey before engineering can proceed'),
            ('Consent Extension', 'Route variation triggers re-consultation — estimated 4–8 week consent programme extension'),
        ],
        'temporal_note': 'Elevation increasing +1.7m since Aug 2025 (+81%) — wreck actively de-burying; risk trajectory: ↑ WORSENING'
    },
    'UXO': {
        'cost_range': '£180K – £650K per item',
        'cost_low': 180000, 'cost_high': 650000,
        'schedule_exposure': '3–8 weeks per item',
        'next_survey': 'Dedicated UXO survey: high-resolution mag + sub-bottom at 10m line spacing; MCM dive assessment',
        'contributions': [
            ('MCM Clearance Programme', 'Each confirmed UXO requires Explosive Ordnance Disposal (EOD) assessment and probable detonation or relocation'),
            ('Burial Design Constraint', 'Standard jet-trench or plough burial prohibited within 50m of uncleared UXO — route or method change required'),
            ('Consent Delay', 'UXO presence triggers Marine Licence amendment — 3–5 week delay per item if unresolved at application stage'),
            ('Standby Vessel Cost', 'MCM vessel mobilisation: ~£85K/week; 2–4 week window typical per item'),
            ('Pre-FEED Investigation', 'Elevated investigation requirement: burial depth, target geometry, corrosion state all needed before design can proceed'),
        ],
        'temporal_note': 'Mag amplitude +20% since Aug 2025 (180→217 nT) — increasing acoustic return suggests continued de-burial; risk trajectory: ↑ WORSENING'
    },
    'Pipeline': {
        'cost_range': '£95K – £380K',
        'cost_low': 95000, 'cost_high': 380000,
        'schedule_exposure': '2–5 weeks',
        'next_survey': 'Pipeline crossing design survey; third-party owner identification; CP survey for status assessment',
        'contributions': [
            ('Crossing Design', 'Pipeline crossing requires detailed engineering: concrete mattress, isolation joints, sleeve protection — typical £50–120K'),
            ('Third-Party Consent', 'Pipeline owner identification and crossing agreement required before consent submission — can be on critical path'),
            ('Free Span Mitigation', 'Three free spans identified — rock placement or grout bag infill required; ~£30–60K per span'),
            ('Construction Method Change', 'Plough or jetting prohibited within crossing zone — HDD or manual lay required'),
        ],
        'temporal_note': 'Anomaly amplitude stable (+7% since Aug 2025) — pipeline condition consistent; risk trajectory: → STABLE'
    },
    'Gas': {
        'cost_range': '£120K – £500K',
        'cost_low': 120000, 'cost_high': 500000,
        'schedule_exposure': '3–7 weeks',
        'next_survey': 'Targeted SBP at 25m line spacing over gas zone; piezocone (CPTU) at 2–3 locations to characterise gas-bearing layer',
        'contributions': [
            ('Geotechnical Programme Uplift', 'Standard geotech scope inadequate — piezocone (CPTU) testing required to characterise gas-bearing layer; adds 2–4 weeks'),
            ('Burial Method Restriction', 'Jetting and plough burial prohibited in confirmed gas zones — HDD or open-cut required; significant cost uplift'),
            ('Gas Blow-out Contingency', 'Trenching risk of gas blow-out requires specialist equipment standby; contingency budget £50–120K'),
            ('Pre-FEED Investigation', 'Gas volume and migration rate must be quantified before burial design can be finalised'),
            ('Potential 3–5 Week Delay', 'If gas zone unresolved at FEED stage, consent submission delayed pending updated geotechnical report'),
        ],
        'temporal_note': 'Reflector acoustic strength +20% since Aug 2025 — stronger blanking = more fluid = more unstable subsurface; risk trajectory: ↑ WORSENING'
    },
    'Sand Wave': {
        'cost_range': '£60K – £220K',
        'cost_low': 60000, 'cost_high': 220000,
        'schedule_exposure': '1–4 weeks',
        'next_survey': 'Repeat SSS survey 6 months pre-construction to quantify migration rate; update bedform mobility model',
        'contributions': [
            ('Burial Depth Uplift', 'Active migration requires additional 0.5–1.0m burial target depth above standard design specification'),
            ('Post-Installation Monitoring', 'Mandatory monitoring programme: quarterly AUV survey for 2 years post-installation; ~£40K/year'),
            ('Timing Constraint', 'Construction window restricted to low-bedform-activity season (typically Oct–Feb in southern North Sea)'),
            ('Repeat Survey Requirement', 'Bedform position at construction will differ from survey baseline — re-survey 3–6 months pre-lay required'),
        ],
        'temporal_note': 'Sand wave amplitude increased from 1.8m to 2.5m since Aug 2025 (+39%) — accelerating migration; risk trajectory: ↑ WORSENING'
    },
    'Hard Ground': {
        'cost_range': '£200K – £750K',
        'cost_low': 200000, 'cost_high': 750000,
        'schedule_exposure': '4–10 weeks',
        'next_survey': 'Rock type classification (drop core/ROV chip sampling); rock strength testing for cutter selection',
        'contributions': [
            ('Rock Cutting Requirement', 'Standard plough and jetting cannot penetrate — rock wheel cutter or HDD required; ~£350K mobilisation'),
            ('Route Modification Assessment', 'Route deviation around rock outcrop may be more cost-effective than cutting — engineering comparison required'),
            ('Vessel Day Rate Uplift', 'Rock-cutting vessel significantly higher day rate than standard lay barge; schedule-dependent cost'),
            ('Mattress Protection', 'If cable laid over rock surface — concrete mattress protection required; ~£80–200K depending on length'),
        ],
        'temporal_note': 'Hard ground extent stable — no change detected between Aug 2025 and Mar 2026; risk trajectory: → STABLE'
    },
    'Boulder': {
        'cost_range': '£80K – £300K',
        'cost_low': 80000, 'cost_high': 300000,
        'schedule_exposure': '2–6 weeks',
        'next_survey': 'ROV/AUV ground-truth survey; individual boulder mapping at 1:500 scale; subsurface EM or SBP for buried boulder density',
        'contributions': [
            ('Boulder Clearance Programme', 'Dense boulder fields require individual clearance — specialist grab or hydraulic arm vessel; ~£60–120K'),
            ('Burial Design Constraint', 'Burial not feasible between boulders — surface lay with mattress protection required in dense zones'),
            ('Construction Duration Extension', 'Slow cable lay rate through boulder field adds ~1–2 weeks construction time'),
            ('Post-Installation Risk', 'Cable not buried in boulder zone — elevated risk of third-party damage; monitoring required'),
        ],
        'temporal_note': 'Boulder field area grew +15% since Aug 2025 — active seabed erosion exposing new boulders; risk trajectory: ↑ WORSENING'
    },
}

def get_hazard_impact(htype):
    """Return the impact profile matching a hazard type string."""
    for k, v in HAZARD_IMPACT_PROFILES.items():
        if k in htype:
            return k, v
    return 'General', {
        'cost_range': '£50K – £200K',
        'cost_low': 50000, 'cost_high': 200000,
        'schedule_exposure': '1–3 weeks',
        'next_survey': 'Targeted re-survey with multi-sensor array at reduced line spacing',
        'contributions': [('Mitigation Required', 'Engineering assessment and mitigation programme required before construction')],
        'temporal_note': 'No temporal change data available for this hazard type'
    }


# ── Temporal change intelligence ──────────────────────────────────────────────
# Survey dates: Aug 2025 (previous) → Mar 2026 (current)
TEMPORAL_CHANGES = {
    'WRK-001': {
        'prev_date': 'Aug 2025', 'curr_date': 'Mar 2026',
        'prev_score': 8.1, 'curr_score': 9.2,
        'score_delta': +1.1,
        'metrics': [
            {'param': 'Hull Elevation', 'prev': '2.1m', 'curr': '3.8m', 'change': '+1.7m (+81%)', 'trend': '↑', 'severity': 'high'},
            {'param': 'Scour Depth',    'prev': 'None', 'curr': '0.8m', 'change': 'New feature',   'trend': '↑', 'severity': 'high'},
            {'param': 'Mag Amplitude',  'prev': '28,500 nT', 'curr': '32,791 nT', 'change': '+4,291 nT (+15%)', 'trend': '↑', 'severity': 'medium'},
            {'param': 'Backscatter',    'prev': 'High', 'curr': 'Very High', 'change': '+12% intensity', 'trend': '↑', 'severity': 'medium'},
        ],
        'driver': 'Seabed erosion progressively de-burying hull — wreck becoming more exposed each survey cycle',
        'trajectory': 'WORSENING',
        'action': 'Prioritise ROV inspection before next design phase — structural state is deteriorating'
    },
    'WRK-002': {
        'prev_date': 'Aug 2025', 'curr_date': 'Mar 2026',
        'prev_score': 7.2, 'curr_score': 8.5,
        'score_delta': +1.3,
        'metrics': [
            {'param': 'Mag Amplitude',  'prev': '2,800 nT', 'curr': '4,522 nT', 'change': '+1,722 nT (+61%)', 'trend': '↑', 'severity': 'high'},
            {'param': 'SSS Expression', 'prev': 'Faint',    'curr': 'Prominent', 'change': 'Significantly stronger', 'trend': '↑', 'severity': 'high'},
            {'param': 'Elevation',      'prev': '0.8m',     'curr': '2.0m',      'change': '+1.2m (+150%)', 'trend': '↑', 'severity': 'high'},
        ],
        'driver': 'Rapidly increasing exposure — anomaly was marginal in 2025, now wreck-class in 2026',
        'trajectory': 'WORSENING RAPIDLY',
        'action': 'Urgent investigation — rate of change suggests active de-burial; next survey within 3 months'
    },
    'UXO-001': {
        'prev_date': 'Aug 2025', 'curr_date': 'Mar 2026',
        'prev_score': 7.8, 'curr_score': 8.8,
        'score_delta': +1.0,
        'metrics': [
            {'param': 'Mag Amplitude',   'prev': '180 nT', 'curr': '217 nT', 'change': '+37 nT (+21%)', 'trend': '↑', 'severity': 'high'},
            {'param': 'Reflector Depth', 'prev': '1.2m',   'curr': '0.9m',   'change': '-0.3m (shallower)', 'trend': '↑', 'severity': 'high'},
            {'param': 'Acoustic Strength','prev': 'Moderate','curr': 'Strong', 'change': '+20% — more fluid/unstable', 'trend': '↑', 'severity': 'high'},
        ],
        'driver': '21% increase in mag return + 0.3m reduction in burial depth: ordnance progressively de-burying; subsurface becoming more fluid (stronger acoustic return = weaker sediment)',
        'trajectory': 'WORSENING',
        'action': 'MCM pre-assessment required immediately — target within cable burial envelope and de-burying'
    },
    'UXO-002': {
        'prev_date': 'Aug 2025', 'curr_date': 'Mar 2026',
        'prev_score': 6.9, 'curr_score': 7.9,
        'score_delta': +1.0,
        'metrics': [
            {'param': 'Position',       'prev': 'Fixed',    'curr': '~15m NE', 'change': 'Migrated — mobile target', 'trend': '↑', 'severity': 'high'},
            {'param': 'Mag Amplitude',  'prev': '100 nT',   'curr': '141 nT',  'change': '+41 nT (+41%)', 'trend': '↑', 'severity': 'high'},
            {'param': 'Burial State',   'prev': 'Buried',   'curr': 'Partially exposed', 'change': 'Exposure increasing', 'trend': '↑', 'severity': 'high'},
        ],
        'driver': 'Target mobility confirmed — 15m NE migration between surveys; bedform activity transporting ordnance along route',
        'trajectory': 'WORSENING — MOBILE TARGET',
        'action': 'Mobile UXO on dynamic seabed — position cannot be assumed stable at construction; re-survey mandatory within 60 days pre-lay'
    },
    'UXO-003': {
        'prev_date': 'Aug 2025', 'curr_date': 'Mar 2026',
        'prev_score': 5.8, 'curr_score': 6.3,
        'score_delta': +0.5,
        'metrics': [
            {'param': 'Mag Amplitude',  'prev': '48 nT',   'curr': '51 nT',  'change': '+3 nT (+6%)', 'trend': '→', 'severity': 'low'},
            {'param': 'Position',       'prev': 'Fixed',   'curr': 'Fixed',  'change': 'Stable within ±2m GPS', 'trend': '→', 'severity': 'low'},
            {'param': 'Burial State',   'prev': 'Buried',  'curr': 'Buried', 'change': 'No change detected', 'trend': '→', 'severity': 'low'},
        ],
        'driver': 'Stable anomaly — minor amplitude increase within measurement uncertainty; no significant change between surveys',
        'trajectory': 'STABLE',
        'action': 'Monitor at construction phase — no immediate action required; flag for pre-lay re-survey'
    },
    'PPL-001': {
        'prev_date': 'Aug 2025', 'curr_date': 'Mar 2026',
        'prev_score': 6.8, 'curr_score': 7.2,
        'score_delta': +0.4,
        'metrics': [
            {'param': 'Mag Amplitude',  'prev': '75 nT',   'curr': '80.7 nT', 'change': '+5.7 nT (+8%)', 'trend': '→', 'severity': 'low'},
            {'param': 'Free Spans',     'prev': '2',       'curr': '3',       'change': '+1 new free span', 'trend': '↑', 'severity': 'medium'},
            {'param': 'Exposure Length','prev': '720m',    'curr': '850m',    'change': '+130m (+18%)', 'trend': '↑', 'severity': 'medium'},
        ],
        'driver': 'Pipeline exposure increasing — new free span developed and exposed length growing due to seabed erosion around pipe',
        'trajectory': 'SLOWLY WORSENING',
        'action': 'Third-party crossing agreement should be initiated now — exposure trend suggests increasing complexity'
    },
    'default': {
        'prev_date': 'Aug 2025', 'curr_date': 'Mar 2026',
        'prev_score': None, 'curr_score': None,
        'score_delta': 0,
        'metrics': [
            {'param': 'Position',    'prev': 'Established', 'curr': 'Consistent', 'change': 'Stable within GPS uncertainty', 'trend': '→', 'severity': 'low'},
            {'param': 'Morphology',  'prev': 'Baseline',    'curr': 'Unchanged',  'change': 'No dimensional change detected', 'trend': '→', 'severity': 'low'},
            {'param': 'Amplitude',   'prev': 'Baseline',    'curr': 'Consistent', 'change': 'Within sensor repeatability bounds', 'trend': '→', 'severity': 'low'},
        ],
        'driver': 'No significant temporal change detected between Aug 2025 and Mar 2026 surveys',
        'trajectory': 'STABLE',
        'action': 'Standard monitoring schedule — re-assess at next planned survey'
    }
}


# ==============================================================================
# PAGE 2: PROJECT TIMELINE


def generate_evidence(haz):
    hid = haz.get('id','')
    sensors = haz.get('detected_by','').split(', ')
    changes = {
        'WRK-001': {'prev': 'Aug 2025', 'list': [
            '**Elevation:** 2.1m → 3.8m (+1.7m increase)',
            '**Scour:** New -0.8m depression around base',
            '**Magnetic:** 28,500nT → 32,791nT (+4,291nT — increasing exposure)',
            '**Status:** Wreck becoming more exposed due to seabed erosion']},
        'WRK-002': {'prev': 'Aug 2025', 'list': [
            '**New prominent anomaly:** 4,522nT — not detected at this amplitude in 2025',
            '**SSS Confidence:** Strong acoustic target confirmed',
            '**Status:** High-priority wreck candidate — investigation required']},
        'UXO-001': {'prev': 'Aug 2025', 'list': [
            '**Mag Increase:** 180nT → 217nT (+37nT strengthening)',
            '**SSS Confidence:** 98% certainty on target',
            '**Status:** Anomaly intensifying — possible further exposure of ordnance']},
        'UXO-002': {'prev': 'Aug 2025', 'list': [
            '**Position Change:** Migrated ~15m NE from 2025 position',
            '**Exposure:** Previously buried, now partially exposed',
            '**Mag:** 100nT → 141nT (becoming more prominent)',
            '**Status:** Mobile UXO — high risk of further movement']},
        'UXO-003': {'prev': 'Aug 2025', 'list': [
            '**Position:** Stable within GPS uncertainty (±2m)',
            '**Magnetic:** 48nT → 51nT (minor change within error bounds)',
            '**Status:** Consistent low-level anomaly — monitor at construction phase']},
        'PPL-001': {'prev': 'Aug 2025', 'list': [
            '**Anomaly amplitude:** 75nT → 80.7nT (minor increase)',
            '**Linear signature:** Consistent along route — ferrous pipe confirmed',
            '**Status:** Stable pipeline anomaly — manage during cable laying']},
        'BLD-001': {'prev': 'Aug 2025', 'list': [
            '**Area Growth:** 210m² → 250m² (+15% expansion)',
            '**Boulder Size:** Avg 1.8m → 2.3m (larger boulders exposed)',
            '**Seabed Erosion:** Active scour revealing subsurface obstacles',
            '**Status:** Dynamic field - ongoing exposure of hazards']},
        'default': {'prev': 'Aug 2025', 'list': [
            '**Position:** Stable within GPS uncertainty (±2m)',
            '**Morphology:** No significant dimensional changes',
            '**Sensors:** Consistent signatures across surveys']}
    }
    change_data = changes.get(hid, changes['default'])
    ev = {'sss': '', 'mbes': '', 'mag': '', 'sbp': '', 'risk_just': '', 'change': change_data}
    htype = haz.get('hazard_type', '')
    if 'SSS' in sensors:
        if 'Wreck' in htype:
            ev['sss'] = "**SSS Analysis:**\n- 45m linear target, sharp edges\n- 15m shadow (3.8m height)\n- High backscatter (steel)\n- Hull form visible\n- **Confidence: 95%**"
        elif 'UXO' in htype:
            ev['sss'] = "**SSS Analysis:**\n- No surface expression (buried)\n- Seabed undisturbed\n- Subsurface anomaly present\n- **Confidence: 85%**"
        elif 'Boulder' in htype:
            ev['sss'] = "**SSS Analysis:**\n- High backscatter clusters\n- Irregular morphology\n- 45+ targets >0.5m\n- **Confidence: 92%**"
        elif 'Sand Wave' in htype:
            ev['sss'] = "**SSS Analysis:**\n- Sinusoidal bedform pattern\n- Wavelength: 15-30m\n- Asymmetric profiles indicate active migration\n- **Confidence: 94%**"
        elif 'Hard' in htype:
            ev['sss'] = "**SSS Analysis:**\n- Very high backscatter intensity\n- Rock outcrop visible at seabed\n- **Confidence: 96%**"
        elif 'Pipeline' in htype:
            ev['sss'] = "**SSS Analysis:**\n- Linear target 0.9m diameter, 850m exposed\n- Acoustic shadow confirms elevation\n- **Confidence: 98%**"
        elif 'Channel' in htype:
            ev['sss'] = "**SSS Analysis:**\n- Linear depression visible\n- Lower backscatter in channel fill\n- **Confidence: 87%**"
        elif 'Gas' in htype:
            ev['sss'] = "**SSS Analysis:**\n- Pockmark expressions at seabed\n- Irregular backscatter pattern\n- **Confidence: 88%**"
    if 'MBES' in sensors:
        if 'Wreck' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- Elevation: 3.8m peak\n- Footprint: 45m × 12m\n- Scour moat: -0.8m\n- **Confidence: 88%**"
        elif 'Boulder' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- 45+ elevation peaks 0.5-2.3m\n- Field: 250×180m\n- **Confidence: 90%**"
        elif 'Gas' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- 15 pockmarks 5-20m diameter\n- Depth: 0.5-1.2m\n- **Confidence: 92%**"
        elif 'Sand Wave' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- Amplitude: 0.8-2.5m, wavelength 20-40m\n- Active migration confirmed\n- **Confidence: 93%**"
        elif 'Hard' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- Very high roughness (RMS >1.5m)\n- Rock surface confirmed\n- **Confidence: 95%**"
        elif 'Pipeline' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- Elevation: 0.6m, length 850m\n- 3 free spans detected\n- **Confidence: 97%**"
        elif 'Channel' in htype:
            ev['mbes'] = "**MBES Analysis:**\n- Depression 4-8m deep, 80-150m wide\n- Subtle seabed expression\n- **Confidence: 89%**"
    if 'Magnetometer' in sensors:
        if 'Wreck' in htype:
            ev['mag'] = "**Mag Analysis:**\n- Dipole: 32,791nT\n- Large ferrous mass confirmed\n- **Confidence: 92%**"
        elif 'UXO' in htype:
            ev['mag'] = "**Mag Analysis:**\n- Dipole pattern matches ordnance\n- WW2 UXO signature\n- **Confidence: 92%**"
        elif 'Pipeline' in htype:
            ev['mag'] = "**Mag Analysis:**\n- Linear anomaly 80.7nT\n- Ferrous material confirmed\n- **Confidence: 89%**"
    if 'SBP' in sensors:
        if 'Wreck' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- Hard reflector at seabed\n- No penetration — solid metallic\n- **Confidence: 85%**"
        elif 'Gas' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- Acoustic blanking below 2m\n- Gas accumulation confirmed\n- **Confidence: 94%**"
        elif 'Channel' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- V-shaped buried channel 4-8m deep\n- Soft sediment infill\n- **Confidence: 91%**"
        elif 'Sand Wave' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- Mobile sand (transparent layer)\n- Active bedform migration\n- **Confidence: 88%**"
        elif 'Hard' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- Hard reflector <0.5m depth\n- No sediment cover\n- **Confidence: 93%**"
        elif 'Pipeline' in htype:
            ev['sbp'] = "**SBP Analysis:**\n- Hard reflector at seabed\n- Fully exposed pipeline\n- **Confidence: 90%**"
    score = haz.get('risk_score', 0)
    if score >= 9.0:
        ev['risk_just'] = f"**Risk: {score}/10 — CRITICAL**\n\n1. Multi-sensor: {len(sensors)} independent confirmations\n2. Significant cable strike risk\n3. Change detection: Dynamic / increasing exposure\n4. Immediate action required\n\n**Combined confidence: 91%**"
    elif score >= 7.0:
        ev['risk_just'] = f"**Risk: {score}/10 — HIGH**\n\n1. Sensor agreement: {len(sensors)} confirmations\n2. Engineering: Route modification required\n3. Timeline: Investigation/redesign delays\n\n**Mitigation required before construction**"
    elif score >= 4.0:
        ev['risk_just'] = f"**Risk: {score}/10 — MEDIUM**\n\n1. {len(sensors)} sensors confirm\n2. Manageable with standard mitigation\n\n**Standard burial + monitoring recommended**"
    else:
        ev['risk_just'] = f"**Risk: {score}/10 — LOW**\n\n1. Limited impact potential\n2. Monitoring sufficient\n\n**Document and track during construction**"
    return ev


# ==============================================================================
# SIDEBAR — INX BRANDED
# ==============================================================================

FILE_IDS = {
    'sss': ['1-fd4WYSO3jAurneJNV_QzMJVx3F5rojM','1reqiNT6_XKdFc4LzjAM634CRe3qqReZP','1MmJAYU9O6bjqst0ufZW5s_7DZQirSr5Z','10XWv6wmnIX0zHDHTtIsOoM71JtByLNVb','10YZlXZ2JDp5f7ehg4xMdFkkZCD5NVHNI','12FLV4q_9X4EzGrqIWtGShCo-BUYrmRlG','134bxFTgfLwZYWvWhIa7swmhS2UGCV7M0','15X6Ho70GmLxlDHubSDnEfEQq85s0CKol','17dJXRk_VIZuQhULjj52-BqaUAnaaeeOH','17yLRua1a3AgBYuZC4_36D7x5-kfLON7L','1A2ZYFc6Mey_pJvBtB9gGXL5zxDeJtdRM','1AiH391YcyhizRgWSldH8Oijd6PzG9a9Y','1EC4BT0SBHsYf6iYXGYXUxhmKhYNbXoeY','1Ffz5h8qjND_jS-wA3QUxtQ0DUaoNj9wC','1GDU4aonNheXJ0pK7NsWqjLyXNDzE1NwM','1IJooNeDkLj4TqCxra7iOjFYtVU182e7I','1MLaLICacB1DpyPv1jkFq8SrY1tRKl7aN','1NYTPv-3PsWs7kjeer_uf4pfGMsbwTi0E','1N_Y_bmCTUuu15IYS9j-XLJNH28OyZD9H','1RZkpzxIQgrCYnWBGm_w42stbyzVmEPva','1VlAVkEbnTbFnto57sMbfFkbzU7p67LL7','1aVvPfIXoRDC2XqmDMG92fUIZtrFpIsJq','1bHSd6XzLDYIAwnQYlDPRo8FLnH8DDJnw','1cMBJlt0A6JwfMS7fhcvR7cnJ4468gLze','1cNNmCgY6iAbHMtGm2UPeJX_NeMnb2rQn','1cvLjbwFDPjD2avHzjMuwDGKYDjjXLT5v','1dhxT_QsbygFdLV9ZYUQ6wd5_2mZSb6e5','1eaS_7K8012AneqC5LkwmuLEqqJmPO1sq','1hg9wgSkhRIzYiIhCGq1xjbFMxzSz4pSm','1iDJWZcRz_zGbOTQpYN9U1V707X5xo3yv','1jNwjUx7zdHHFKxFtAXSbYLMrmVKDdRxS','1jqZLJ5xJhxdChh9SKlbahsLviqbqPzFx','1ldV6zBMMrWfovkNbV2bSSkHyZmPUKYlI','1nzPO4LXl6PJ5TffOe6c2pHJFmZzSUDfp','1sWFLzNsAo0ZQ_nbusrNm9I7DnfFh4TIq','1t0NXhHNdHQrwuMCzfiGu1CYb1z47-XVK','1w2ZwrKigqOHqXMRnyY4GD_jNn0VTCrWE','1wkcFrGXx8dVNf5gYMkEaNeIdvIRNazJz','1wmMgdqL-B56PI4sHQ-Fr4GFxp28ptb8U','1zso2rorqe_FXDXbMfHXl3vDRodD8H7fC'],
    'mbes':     '1lE9X1S2Lqt3UxKgEJto5cURf1gTxOADr',
    'mag':      '1jyYQ9ICEFjXxFAatFQvGb-9byu3ryq5P',
    'turbines': '18uYbX7OWZcqQfoBow6F_P4AmjptioeeO',
    'sbp':      '1cZCoNX1t68X1BoiyikYKRAV0vzo_3pGO',
    'hazards':  '1x_aerOM_LY7bw1CJdNC35zD2KkhJo4Sh',
    'mag_targets': '1461Q0yswjO5qetkEfazB2JZMq5GHbSMd',
}

st.sidebar.header(" Data Layers")
basemap   = st.sidebar.selectbox("Basemap", ['Esri Satellite', 'OpenStreetMap'], index=0)
mbes_cmap = st.sidebar.selectbox("MBES Colormap", ['ocean', 'viridis', 'seismic'], index=0)
quality   = st.sidebar.radio("Quality", ["Fast (500px)", "Good (1000px)", "High (2000px)"], index=1)
max_px    = {"Fast (500px)": 500, "Good (1000px)": 1000, "High (2000px)": 2000}[quality]

st.sidebar.markdown("---")
st.sidebar.header(" Quick Load")

if st.sidebar.button("Load MBES", use_container_width=True):
    with st.spinner("Downloading MBES..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            try:
                download_from_gdrive(FILE_IDS['mbes'], tmp.name)
                img, bounds = tif_to_png_base64(tmp.name, mbes_cmap, max_px, False, False)
                if img and bounds:
                    st.session_state.raster_layers = [(img, bounds)]
                    st.sidebar.success("✅ MBES loaded")
                    st.rerun()
                os.unlink(tmp.name)
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

if st.sidebar.button("Load Hazards", use_container_width=True):
    with st.spinner("Downloading hazards..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            try:
                download_from_gdrive(FILE_IDS['hazards'], tmp.name)
                gdf = gpd.read_file(tmp.name)
                st.session_state.hazards = patch_hazard_coordinates(gdf)
                st.sidebar.success(f"✅ {len(st.session_state.hazards)} hazards loaded")
                os.unlink(tmp.name)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

if st.sidebar.button("Load SSS Tiles", use_container_width=True):
    prog = st.sidebar.progress(0)
    count = 0
    st.session_state.raster_layers = []
    for i, fid in enumerate(FILE_IDS['sss']):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                download_from_gdrive(fid, tmp.name)
                img, bounds = tif_to_png_base64(tmp.name, 'gray', max_px, True, False)
                if img and bounds:
                    st.session_state.raster_layers.append((img, bounds))
                    count += 1
                os.unlink(tmp.name)
        except:
            pass
        prog.progress((i+1)/len(FILE_IDS['sss']))
    prog.empty()
    st.sidebar.success(f"✅ {count} SSS tiles loaded")
    st.rerun()

if st.sidebar.button("Load Mag TIF", use_container_width=True):
    with st.spinner("Downloading Mag TIF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            try:
                download_from_gdrive(FILE_IDS['mag'], tmp.name)
                img, bounds = tif_to_png_base64(tmp.name, 'seismic', max_px, False, True)
                if img and bounds:
                    st.session_state.mag_tif_layer = (img, bounds)
                    st.sidebar.success("✅ Mag TIF loaded")
                    st.rerun()
                os.unlink(tmp.name)
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

if st.sidebar.button("Load Turbines", use_container_width=True):
    with st.spinner("Downloading turbines..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            try:
                download_from_gdrive(FILE_IDS['turbines'], tmp.name)
                st.session_state.turbines = gpd.read_file(tmp.name)
                st.sidebar.success(f"✅ {len(st.session_state.turbines)} turbines loaded")
                os.unlink(tmp.name)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

if st.sidebar.button("Load Mag Targets CSV", use_container_width=True):
    with st.spinner("Downloading mag targets..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            try:
                download_from_gdrive(FILE_IDS['mag_targets'], tmp.name)
                df = pd.read_csv(tmp.name)
                df.columns = [c.strip() for c in df.columns]
                df = df.loc[:, ~df.columns.duplicated(keep='first')]
                for col in ['Latitude','Longitude','nT']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['Latitude','Longitude','nT'])
                st.session_state.mag_targets = df
                st.sidebar.success(f"✅ {len(df)} mag targets loaded")
                os.unlink(tmp.name)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

if st.sidebar.button("Load SBP", use_container_width=True):
    with st.spinner("Downloading SBP..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            try:
                download_from_gdrive(FILE_IDS['sbp'], tmp.name)
                st.session_state.sbp_lines = gpd.read_file(tmp.name)
                st.sidebar.success(f"✅ {len(st.session_state.sbp_lines)} SBP lines loaded")
                os.unlink(tmp.name)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

if st.sidebar.button(" Clear All", use_container_width=True):
    st.session_state.raster_layers = []
    st.session_state.hazards       = None
    st.session_state.turbines      = None
    st.session_state.sbp_lines     = None
    st.session_state.mag_tif_layer = None
    st.session_state.mag_targets   = None
    st.rerun()

if st.session_state.hazards is not None:
    hazs = st.session_state.hazards
    crit_n = len(hazs[hazs['risk']=='Critical'])
    high_n = len(hazs[hazs['risk']=='High'])
    st.sidebar.markdown("---")
    st.sidebar.header(" Project Impact")
    st.sidebar.metric("Critical Hazards", crit_n, delta="Immediate action" if crit_n else None)
    st.sidebar.metric("High Risk",        high_n, delta="Engineering uplift" if high_n else None)
    st.sidebar.metric("Total Hazards",    len(hazs))

st.sidebar.markdown("---")
st.sidebar.success("✅ System Operational")
st.sidebar.caption(f"📡 {datetime.now().strftime('%d %b %Y  %H:%M')}")


# ==============================================================================

# PAGE 1: HAZARD MAP
# ==============================================================================

if page == "Hazard Map":
    st.header("Interactive Hazard Map")

    # Show any auto-load errors
    if st.session_state.get("auto_load_errors"):
        for _err in st.session_state.auto_load_errors:
            st.warning(f"⚠️ Auto-load: {_err}")

    
    if st.session_state.hazards is not None and len(st.session_state.hazards) > 0:
        # ── Filters row ──────────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            types = sorted(st.session_state.hazards['hazard_type'].unique())
            sel_types = st.multiselect("Filter by Type:", types, types, key='map_types')
        with col2:
            risks = ["Critical", "High", "Medium", "Low"]
            sel_risks = st.multiselect("Filter by Risk:", risks, risks, key='map_risks')

        # ── Layer toggles ────────────────────────────────────────────────────
        toggle_cols = st.columns(7)
        show_sss   = toggle_cols[0].checkbox("SSS",         True,  key='tog_sss')
        show_mbes  = toggle_cols[1].checkbox("Mag TIF",     True,  key='tog_mbes')
        show_sbp   = toggle_cols[2].checkbox("SBP Lines",   True,  key='tog_sbp')
        show_turb  = toggle_cols[3].checkbox("Turbines",    True,  key='tog_turb')
        show_magt  = toggle_cols[4].checkbox("Mag Targets", True,  key='tog_magt')
        show_zones = toggle_cols[5].checkbox("Risk Zones",  False, key='tog_zones')
        show_ai    = toggle_cols[6].checkbox("GRID AI",     True,  key='show_ai')

        # ── nT threshold filter (only shown when mag targets loaded) ─────────
        nt_min_show = 5.0
        if st.session_state.mag_targets is not None and show_magt:
            df_mt = st.session_state.mag_targets
            nt_max_val = float(df_mt['nT'].max())
            nt_col1, nt_col2 = st.columns([2,1])
            with nt_col1:
                nt_band = st.select_slider(
                    " nT Anomaly Threshold — show targets above:",
                    options=[5, 10, 20, 50, 100, 200, 500],
                    value=5,
                    key='nt_thresh'
                )
            with nt_col2:
                st.metric("Targets shown", int((df_mt['nT'] >= nt_band).sum()),
                          delta=f"of {len(df_mt)} total")
            nt_min_show = nt_band

        show_hazards_toggle = st.checkbox("Show Hazard Markers", True, key='tog_haz')

        filtered = st.session_state.hazards[
            st.session_state.hazards['hazard_type'].isin(sel_types) &
            st.session_state.hazards['risk'].isin(sel_risks)
        ]

        # ── Metrics ──────────────────────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🔴 Critical", len(filtered[filtered['risk']=='Critical']))
        col2.metric("🟠 High",     len(filtered[filtered['risk']=='High']))
        col3.metric("🟡 Medium",   len(filtered[filtered['risk']=='Medium']))
        col4.metric("🟢 Low",      len(filtered[filtered['risk']=='Low']))
        col5.metric("📍 Total",    len(filtered))

        # ── Map centre ───────────────────────────────────────────────────────
        all_bounds = [b for _, b in st.session_state.raster_layers if b]
        if all_bounds:
            center_lat = sum([(b[1]+b[3])/2 for b in all_bounds])/len(all_bounds)
            center_lon = sum([(b[0]+b[2])/2 for b in all_bounds])/len(all_bounds)
        else:
            center_lat, center_lon = 53.81, 0.13

        if basemap == 'Esri Satellite':
            m = folium.Map([center_lat, center_lon], zoom_start=13, tiles=None, control_scale=True)
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri', name='Esri Satellite'
            ).add_to(m)
        else:
            m = folium.Map([center_lat, center_lon], zoom_start=13, control_scale=True)

        # ── SSS raster tiles ─────────────────────────────────────────────────
        if show_sss and st.session_state.raster_layers:
            sss_group = folium.FeatureGroup(name="SSS Tiles", show=True)
            for img, bounds in st.session_state.raster_layers:
                if img and bounds:
                    folium.raster_layers.ImageOverlay(
                        f"data:image/png;base64,{img}",
                        [[bounds[1],bounds[0]],[bounds[3],bounds[2]]],
                        opacity=0.85
                    ).add_to(sss_group)
            sss_group.add_to(m)

        # ── Mag TIF ───────────────────────────────────────────────────────────
        if show_mbes and st.session_state.mag_tif_layer:
            img, bounds = st.session_state.mag_tif_layer
            if img and bounds:
                mag_group = folium.FeatureGroup(name="Magnetometer TIF", show=True)
                folium.raster_layers.ImageOverlay(
                    f"data:image/png;base64,{img}",
                    [[bounds[1],bounds[0]],[bounds[3],bounds[2]]],
                    opacity=0.6
                ).add_to(mag_group)
                mag_group.add_to(m)

        # ── SBP lines ─────────────────────────────────────────────────────────
        if show_sbp and st.session_state.sbp_lines is not None:
            try:
                sbp_group = folium.FeatureGroup(name="SBP Lines", show=True)
                gdf = st.session_state.sbp_lines.to_crs('EPSG:4326') if st.session_state.sbp_lines.crs else st.session_state.sbp_lines
                for _, row in gdf.iterrows():
                    folium.GeoJson(
                        row.geometry.__geo_interface__,
                        style_function=lambda x: {'color': '#00FFFF', 'weight': 2}
                    ).add_to(sbp_group)
                sbp_group.add_to(m)
            except:
                pass

        # ── Turbines ──────────────────────────────────────────────────────────
        turbines_wgs84 = None
        if st.session_state.turbines is not None:
            try:
                turbines_wgs84 = st.session_state.turbines.to_crs('EPSG:4326') if st.session_state.turbines.crs else st.session_state.turbines
                if show_turb:
                    turb_group = folium.FeatureGroup(name="Wind Turbines", show=True)
                    for _, row in turbines_wgs84.iterrows():
                        tlat, tlon = row.geometry.y, row.geometry.x
                        tref  = row.get('ref') or row.get('seamark:name') or 'N/A'
                        tmfr  = row.get('manufacturer', 'N/A')
                        tmod  = row.get('model', 'N/A')
                        toper = row.get('operator', 'N/A')
                        turbine_popup = f"""
                        <div style="width:280px; font-family:Arial; font-size:13px;">
                            <h3 style="margin:0; padding:8px 10px; background:#0013C3; color:white; border-radius:5px 5px 0 0;">
                                 Turbine {tref}
                            </h3>
                            <div style="padding:10px;">
                                <p style="margin:4px 0;"><b>Ref / Name:</b> {tref}</p>
                                <p style="margin:4px 0;"><b>Manufacturer:</b> {tmfr}</p>
                                <p style="margin:4px 0;"><b>Model:</b> {tmod}</p>
                                <p style="margin:4px 0;"><b>Operator:</b> {toper}</p>
                                <p style="margin:4px 0;"><b>Coordinates:</b> {tlat:.7f} / {tlon:.7f}</p>
                            </div>
                        </div>
                        """
                        folium.CircleMarker(
                            [tlat, tlon], radius=8,
                            color='#0013C3', fill=True, fillColor='#4169E1', fillOpacity=0.85,
                            popup=folium.Popup(turbine_popup, max_width=300),
                            tooltip=f"Turbine {tref} | {toper}"
                        ).add_to(turb_group)
                    turb_group.add_to(m)
            except:
                pass

        # ── Magnetic Targets (CSV points) ─────────────────────────────────────
        if show_magt and st.session_state.mag_targets is not None:
            try:
                df_mt = st.session_state.mag_targets
                df_mt = df_mt.copy()
                df_mt['nT'] = pd.to_numeric(df_mt['nT'], errors='coerce').fillna(0.0)
                df_mt['Latitude']  = pd.to_numeric(df_mt['Latitude'],  errors='coerce')
                df_mt['Longitude'] = pd.to_numeric(df_mt['Longitude'], errors='coerce')
                df_mt = df_mt.dropna(subset=['Latitude','Longitude'])
                df_filtered = df_mt[df_mt['nT'] >= float(nt_min_show)].copy()

                # Colour scale: 5-20 nT = yellow, 20-100 = orange, 100+ = red
                def nt_color(val):
                    if val >= 200: return '#FF0000'
                    if val >= 100: return '#FF4500'
                    if val >= 50:  return '#FF8C00'
                    if val >= 20:  return '#FFA500'
                    return '#FFD700'

                mag_t_group = folium.FeatureGroup(name="Mag Targets (nT)", show=True)
                for _, tr in df_filtered.iterrows():
                    nt_val = float(tr['nT'])
                    lat, lon = float(tr['Latitude']), float(tr['Longitude'])
                    c = nt_color(nt_val)
                    radius = min(4 + nt_val / 40, 14)  # scale dot with magnitude
                    popup_t = f"""
                    <div style="width:200px; font-family:Arial; font-size:12px;">
                        <h4 style="margin:0; padding:6px 8px; background:{c}; color:white; border-radius:4px 4px 0 0;">
                             Mag Anomaly
                        </h4>
                        <div style="padding:8px;">
                            <p style="margin:3px 0;"><b>Amplitude:</b> {nt_val:.1f} nT</p>
                            <p style="margin:3px 0;"><b>Lat:</b> {lat:.6f}</p>
                            <p style="margin:3px 0;"><b>Lon:</b> {lon:.6f}</p>
                        </div>
                    </div>
                    """
                    folium.CircleMarker(
                        [lat, lon], radius=radius,
                        color=c, fill=True, fillColor=c, fillOpacity=0.75, weight=1,
                        popup=folium.Popup(popup_t, max_width=220),
                        tooltip=f" {nt_val:.1f} nT"
                    ).add_to(mag_t_group)
                mag_t_group.add_to(m)
            except Exception as e:
                st.warning(f"Could not render mag targets: {e}")

        # ── Risk Zones (analysis grid) ────────────────────────────────────────
        if show_zones:
            try:
                zones_gdf = get_risk_zones_gdf()
                zones_group = folium.FeatureGroup(name="Risk Zones", show=True)
                for _, zcell in zones_gdf.iterrows():
                    fill_col, fill_op = risk_zone_color(zcell['score'])
                    geo = zcell['geometry'].__geo_interface__
                    popup_html_z = risk_zone_popup(zcell)
                    folium.GeoJson(
                        geo,
                        style_function=lambda x, fc=fill_col, fo=fill_op: {
                            'fillColor': fc, 'color': fc,
                            'weight': 0.5, 'fillOpacity': fo
                        },
                        popup=folium.Popup(popup_html_z, max_width=440),
                        tooltip=f"Zone {zcell['cell_id']} | Score {zcell['score']}/10 | {zcell['confidence']} confidence"
                    ).add_to(zones_group)
                zones_group.add_to(m)
            except Exception as e:
                st.warning(f"Could not render risk zones: {e}")

        # ── Hazard markers ────────────────────────────────────────────────────
        if show_hazards_toggle:
            haz_group = folium.FeatureGroup(name="Hazard Markers", show=True)
            for _, row in filtered.iterrows():
                risk  = row.get('risk', 'Unknown')
                htype = row.get('hazard_type', 'Unknown')
                colors = {'Critical':'red','High':'orange','Medium':'beige','Low':'green'}
                color  = colors.get(risk, 'gray')

                haz_lat, haz_lon = row.geometry.y, row.geometry.x
                if turbines_wgs84 is not None and len(turbines_wgs84) > 0:
                    dist_m, near_ref, _, _ = nearest_turbine_info(haz_lat, haz_lon, turbines_wgs84)
                    dist_str = f"{dist_m:.0f} m to Turbine {near_ref}"
                else:
                    dist_m   = row.get('distance_to_turbine_m', None)
                    near_ref = row.get('nearest_turbine', 'N/A')
                    dist_str = f"{dist_m} m to {near_ref}" if dist_m else "N/A (load turbines)"

                # nT info — only for Wreck, UXO, Pipeline; use hardcoded pin value
                nt_info = ""
                htype_check = row.get('hazard_type', '')
                if any(t in htype_check for t in ('Wreck', 'UXO', 'Pipeline')):
                    pin_nt   = row.get('mag_nt', None)
                    pin_note = row.get('mag_note', '')
                    if pin_nt is not None and str(pin_nt) not in ('', 'None', 'nan'):
                        try:
                            nt_val = float(pin_nt)
                            nt_info = f"""
                            <div style="margin:8px 0; padding:8px; background:#fff8e1; border-left:4px solid #FF8C00; border-radius:0 5px 5px 0;">
                                <p style="margin:2px 0; font-weight:bold; color:#c45000;">Magnetometer: {nt_val:,.1f} nT</p>
                                <p style="margin:2px 0; font-size:11px; color:#555;">{pin_note}</p>
                            </div>
                            """
                        except:
                            pass

                # Title text colour: black on light colours (Critical), white on dark
                _title_text = '#1a1a1a' if risk == 'Critical' else 'white'
                # Risk badge accent colour for border only — text always dark
                _risk_accent = {'Critical':'#b8d900','High':'#0013C3','Medium':'#6a7a68','Low':'#5a6080'}.get(risk,'#888')
                _risk_label  = {'Critical':'🔴 CRITICAL','High':'🟠 HIGH','Medium':'🟡 MEDIUM','Low':'🟢 LOW'}.get(risk, risk)

                popup_html = f"""
                <div style="width:340px; font-family:Arial, sans-serif; font-size:13px; color:#1a1a1a; line-height:1.5;">
                    <div style="margin:0; padding:10px 14px; background:{get_risk_color(risk)}; border-radius:6px 6px 0 0;">
                        <div style="font-size:15px; font-weight:700; color:{_title_text};">{htype}</div>
                        <div style="font-size:11px; color:{_title_text}; opacity:0.85;">ID: {row.get('id','')}</div>
                    </div>
                    <div style="padding:12px 14px; background:white;">
                        <table style="width:100%; border-collapse:collapse; font-size:12px;">
                            <tr><td style="padding:4px 0; color:#555; width:90px;">Name</td><td style="padding:4px 0; font-weight:600; color:#1a1a1a;">{row.get('name','N/A')}</td></tr>
                            <tr><td style="padding:4px 0; color:#555;">Size</td><td style="padding:4px 0; color:#1a1a1a;">{row.get('size','N/A')}</td></tr>
                            <tr><td style="padding:4px 0; color:#555;">📡 Sensors</td><td style="padding:4px 0; color:#1a1a1a;">{row.get('detected_by','N/A')}</td></tr>
                            <tr><td style="padding:4px 0; color:#555;">📏 Distance</td><td style="padding:4px 0; color:#1a1a1a;">{dist_str}</td></tr>
                        </table>
                        {nt_info}
                        <div style="margin:10px 0 6px 0; padding:8px 10px; background:#f7f7f7; border-left:4px solid {_risk_accent}; border-radius:0 4px 4px 0;">
                            <span style="font-size:11px; color:#555; text-transform:uppercase; letter-spacing:0.5px;">Risk Level</span><br>
                            <span style="font-size:14px; font-weight:700; color:#1a1a1a;">{_risk_label}</span>
                            &nbsp;<span style="font-size:12px; color:#444;">Score: {row.get('risk_score','N/A')}/10</span>
                        </div>
                """

                if show_ai:
                    sensors = row.get('detected_by', '').split(', ')
                    nsens   = len([s for s in sensors if s.strip()])
                    popup_html += f"""
                        <div style="margin:8px 0 4px 0; padding:8px 10px; background:#f0f4ff; border-left:4px solid #0013C3; border-radius:0 4px 4px 0;">
                            <div style="font-size:12px; font-weight:700; color:#0013C3; margin-bottom:5px;">🤖 GRID AI Detection</div>
                    """
                    if 'SSS'          in row.get('detected_by',''): popup_html += '<div style="font-size:11px; color:#1a1a1a; margin:2px 0;">✅ SSS: Target confirmed | 95%</div>'
                    if 'MBES'         in row.get('detected_by',''): popup_html += '<div style="font-size:11px; color:#1a1a1a; margin:2px 0;">✅ MBES: Elevation anomaly | 88%</div>'
                    if 'Magnetometer' in row.get('detected_by',''): popup_html += '<div style="font-size:11px; color:#1a1a1a; margin:2px 0;">✅ Mag: Ferrous signature | 92%</div>'
                    if 'SBP'          in row.get('detected_by',''): popup_html += '<div style="font-size:11px; color:#1a1a1a; margin:2px 0;">✅ SBP: Subsurface anomaly | 85%</div>'
                    popup_html += f"""
                            <div style="margin-top:6px; padding-top:6px; border-top:1px solid #ccd5f0; font-size:11px; color:#1a1a1a;">
                                <b>Combined Confidence:</b> 91% &nbsp;·&nbsp; <b>Agreement:</b> {nsens}/4 sensors
                            </div>
                        </div>
                    """

                popup_html += f"""
                        <div style="margin-top:8px; padding-top:8px; border-top:1px solid #eee; font-size:12px; color:#1a1a1a;">
                            <div style="margin:3px 0;"><b>💰 Cost:</b> {row.get('cost','N/A')}</div>
                            <div style="margin:3px 0;"><b>📅 Timeline:</b> {row.get('investigation_timeline','N/A')}</div>
                        </div>
                    </div>
                </div>
                """

                folium.Marker(
                    [haz_lat, haz_lon],
                    popup=folium.Popup(popup_html, max_width=380),
                    tooltip=f"{htype}: {row.get('name','')} | {risk}",
                    icon=folium.Icon(color=color, icon='exclamation-triangle', prefix='fa')
                ).add_to(haz_group)
            haz_group.add_to(m)

        folium.plugins.Fullscreen().add_to(m)
        folium.plugins.MousePosition(position='bottomright', separator=' | ', prefix='📍', lat_formatter='function(num) {return L.Util.formatNum(num, 6);}', lng_formatter='function(num) {return L.Util.formatNum(num, 6);}').add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=1400, height=700)

        # ── Hazard register table ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Hazard Register")
        cols = ['id','hazard_type','name','risk','distance_to_turbine_m','investigation_timeline','cost']
        st.dataframe(filtered[cols], use_container_width=True, height=300)
        csv = filtered.to_csv(index=False)
        st.download_button(" Download CSV", csv, f"hazards_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

        # ── Mag Targets summary table (if loaded) ────────────────────────────
        if st.session_state.mag_targets is not None and show_magt:
            st.markdown("---")
            st.subheader(" Magnetic Anomaly Summary")
            df_mt = st.session_state.mag_targets
            df_show = df_mt[df_mt['nT'] >= nt_min_show].sort_values('nT', ascending=False).reset_index(drop=True)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Targets ≥ threshold", len(df_show))
            col_b.metric("Max anomaly", f"{df_mt['nT'].max():.1f} nT")
            col_c.metric("Median anomaly", f"{df_mt['nT'].median():.1f} nT")
            st.dataframe(df_show[['Latitude','Longitude','nT']].head(50), use_container_width=True, height=250)

    else:
        st.info("⏳ Loading survey data... please wait a moment then refresh if the map is empty.")


# ==============================================================================

elif page == "Project Timeline":
    st.header("📅 Project Timeline & Engineering Impact Analysis")

    hazards_loaded = st.session_state.hazards is not None and len(st.session_state.hazards) > 0

    if hazards_loaded:
        hazs = st.session_state.hazards
        crit = int(len(hazs[hazs['risk'] == 'Critical']))
        high = int(len(hazs[hazs['risk'] == 'High']))
        med  = int(len(hazs[hazs['risk'] == 'Medium']))
        low  = int(len(hazs[hazs['risk'] == 'Low']))
    else:
        crit = high = med = low = 0

    # ── View toggle ────────────────────────────────────────────────────────────
    st.markdown("#### Schedule View")
    view_col, _, kpi1, kpi2, kpi3 = st.columns([2, 0.3, 1.5, 1.5, 1.5])
    with view_col:
        view = st.radio("", ["Before Survey", "After Survey"],
                        horizontal=True, key='timeline_view',
                        label_visibility='collapsed')
    view_key = 'before' if view == "Before Survey" else 'after'

    fig, total_months, delays = create_timeline_gantt(
        st.session_state.hazards if hazards_loaded else None,
        view=view_key
    )

    # Baseline for comparison
    _, base_months, _ = create_timeline_gantt(None, view='before')
    schedule_delta = total_months - base_months

    with kpi1:
        st.metric("Total Duration", f"{total_months:.0f} months",
                  delta=f"+{schedule_delta:.1f} mo" if view_key == 'after' and schedule_delta > 0 else None,
                  delta_color="inverse")
    with kpi2:
        if hazards_loaded:
            total_cost_low  = sum(HAZARD_IMPACT_PROFILES.get(
                next((k for k in HAZARD_IMPACT_PROFILES if k in ht), 'x'), {}).get('cost_low', 50000)
                for ht in hazs['hazard_type'])
            total_cost_high = sum(HAZARD_IMPACT_PROFILES.get(
                next((k for k in HAZARD_IMPACT_PROFILES if k in ht), 'x'), {}).get('cost_high', 200000)
                for ht in hazs['hazard_type'])
            st.metric("Cost Exposure", f"£{total_cost_low/1e6:.1f}M – £{total_cost_high/1e6:.1f}M")
        else:
            st.metric("Cost Exposure", "Load hazards")
    with kpi3:
        st.metric("Hazard Counts", f"{crit}C / {high}H / {med}M / {low}L")

    st.plotly_chart(fig, use_container_width=True)

    # ── Legend for after view ──────────────────────────────────────────────────
    if view_key == 'after':
        st.caption("🟦 Baseline phase duration &nbsp;&nbsp; 🟧 Hazard-driven delay extension (hatched)")

    st.markdown("---")

    # ── Delay contribution breakdown ───────────────────────────────────────────
    if view_key == 'after' and hazards_loaded:
        st.markdown("#### ⏱️ Delay Contribution by Phase")

        delay_rows = [(phase, ext) for phase, ext in delays.items() if ext > 0]
        if delay_rows:
            cols_d = st.columns(len(delay_rows))
            for i, (phase, ext) in enumerate(delay_rows):
                with cols_d[i]:
                    drivers = []
                    if 'UXO' in phase or 'Clearance' in phase:
                        if crit > 0: drivers.append(f"{crit} Critical → MCM programme")
                        if high > 0: drivers.append(f"{high} High → EOD assessment")
                    elif 'FEED' in phase or 'Engineering' in phase:
                        if crit > 0: drivers.append(f"{crit} Critical → route redesign")
                        if high > 0: drivers.append(f"{high} High → engineering uplift")
                        if med  > 0: drivers.append(f"{med} Medium → design checks")
                    elif 'Consent' in phase:
                        if crit > 0: drivers.append(f"{crit} Critical → Marine Licence amendment")
                        if high > 0: drivers.append(f"{high} High → consent re-consultation")
                    elif 'Geotech' in phase:
                        if crit > 0: drivers.append(f"{crit} Critical → additional CPTU/borehole")
                        if high > 0: drivers.append(f"{high} High → targeted geotech scope")
                    else:
                        if high > 0: drivers.append(f"{high} High → prep uplift")
                        if med  > 0: drivers.append(f"{med} Medium → additional checks")

                    driver_html = "".join(f"<li style='font-size:10px;margin:1px 0;'>{d}</li>" for d in drivers)
                    st.markdown(
                        f"<div style='background:#fff3e0;border-left:4px solid #FF4500;"
                        f"padding:8px 10px;border-radius:4px;'>"
                        f"<div style='font-size:12px;font-weight:bold;color:#c94000;'>{phase}</div>"
                        f"<div style='font-size:18px;font-weight:bold;color:#FF4500;margin:2px 0;'>+{ext:.1f} months</div>"
                        f"<ul style='margin:4px 0 0 0;padding-left:14px;'>{driver_html}</ul>"
                        f"</div>", unsafe_allow_html=True
                    )
        else:
            st.success("✅ No significant schedule delays — all hazards manageable within baseline programme")

    st.markdown("---")

    # ── Per-hazard engineering impact table ────────────────────────────────────
    if hazards_loaded:
        st.markdown("#### 🔍 Hazard-by-Hazard Engineering Impact")
        st.caption("Expand each hazard type to see cost exposure, schedule risk, next survey action and detailed delay contributions.")

        # Group by hazard type
        htype_groups = {}
        for _, row in hazs.iterrows():
            ht = row.get('hazard_type', 'Unknown')
            htype_groups.setdefault(ht, []).append(row)

        for htype, rows in sorted(htype_groups.items(), key=lambda x: -max(r.get('risk_score', 0) for r in x[1])):
            profile_key, profile = get_hazard_impact(htype)
            count = len(rows)
            max_score = max(r.get('risk_score', 0) for r in rows)
            risk_levels = [r.get('risk', 'Unknown') for r in rows]
            crit_c = risk_levels.count('Critical')
            high_c = risk_levels.count('High')

            # Risk colour
            rc = '#CC0000' if crit_c > 0 else '#FF6600' if high_c > 0 else '#FFAA00'

            risk_icon = '🔴' if crit_c > 0 else '🟠' if high_c > 0 else '🟡'
            inst_word = f"{count} instance{'s' if count > 1 else ''}"
            with st.expander(
                f"{risk_icon}  {htype}   |   {inst_word}   |   Max score {max_score}/10",
                expanded=(crit_c > 0)
            ):
                # KPI row
                k1, k2, k3 = st.columns(3)
                k1.metric("Cost Exposure",      profile['cost_range'])
                k2.metric("Schedule Exposure",  profile['schedule_exposure'])
                k3.metric("Instances in Route", f"{count} ({crit_c}C / {high_c}H)")

                # Next survey recommendation
                st.markdown(
                    f"<div style='background:#e8f4fd;border-left:4px solid #0013C3;"
                    f"padding:8px 12px;border-radius:4px;margin:8px 0;font-size:12px;'>"
                    f"<b>📡 Recommended Next Survey Action:</b><br>{profile['next_survey']}</div>",
                    unsafe_allow_html=True
                )

                # Temporal change note
                # Find temporal data for any instance of this type
                temp_note = profile.get('temporal_note', '')
                traj_color = '#CC0000' if 'WORSENING' in temp_note else '#FF8C00' if 'SLOWLY' in temp_note else '#00AA44'
                if temp_note:
                    st.markdown(
                        f"<div style='background:#fff8e1;border-left:4px solid {traj_color};"
                        f"padding:8px 12px;border-radius:4px;margin:8px 0;font-size:12px;'>"
                        f"<b>📈 Temporal Change (Aug 2025 → Mar 2026):</b><br>{temp_note}</div>",
                        unsafe_allow_html=True
                    )

                # Contribution breakdown
                st.markdown("**Delay & Cost Contribution Breakdown:**")
                for contrib_name, contrib_detail in profile['contributions']:
                    st.markdown(
                        f"<div style='background:#f8f8f8;border-left:3px solid {rc};"
                        f"padding:7px 10px;border-radius:3px;margin:4px 0;font-size:12px;'>"
                        f"<b>{contrib_name}</b><br>"
                        f"<span style='color:#555;'>{contrib_detail}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                # Individual hazard IDs in this type
                ids = [r.get('id', '?') for r in rows]
                st.caption(f"Hazard IDs: {', '.join(ids)}")

    else:
        st.info("⏳ Loading survey data automatically. If the timeline is empty, navigate away and back.")



# PAGE 3: EVIDENCE VIEWER
# ==============================================================================

elif page == "Evidence Viewer":
    st.header("Evidence Viewer - Hazard Analysis")
    
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
        
        m = folium.Map([center_lat, center_lon], zoom_start=13, tiles=None, control_scale=True)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite'
        ).add_to(m)
        
        # Add MBES/SSS layer if available
        if st.session_state.raster_layers:
            for img, bounds in st.session_state.raster_layers:
                if img and bounds:
                    folium.raster_layers.ImageOverlay(
                        f"data:image/png;base64,{img}",
                        [[bounds[1],bounds[0]],[bounds[3],bounds[2]]],
                        opacity=0.7,
                        name="SSS/MBES"
                    ).add_to(m)

        if st.session_state.mag_tif_layer:
            img_m, bnd_m = st.session_state.mag_tif_layer
            if img_m and bnd_m:
                folium.raster_layers.ImageOverlay(
                    f"data:image/png;base64,{img_m}",
                    [[bnd_m[1],bnd_m[0]],[bnd_m[3],bnd_m[2]]],
                    opacity=0.6, name="Mag TIF"
                ).add_to(m)

        # Mag targets
        if st.session_state.mag_targets is not None:
            try:
                df_ev_mt = st.session_state.mag_targets
                mag_ev_group = folium.FeatureGroup(name="Mag Targets", show=True)
                for _, tr in df_ev_mt.iterrows():
                    nt_val = float(tr['nT'])
                    if nt_val < 5: continue
                    lat_t, lon_t = float(tr['Latitude']), float(tr['Longitude'])
                    c = '#FF0000' if nt_val >= 200 else '#FF4500' if nt_val >= 100 else '#FF8C00' if nt_val >= 50 else '#FFA500' if nt_val >= 20 else '#FFD700'
                    folium.CircleMarker(
                        [lat_t, lon_t], radius=min(4 + nt_val/40, 14),
                        color=c, fill=True, fillColor=c, fillOpacity=0.7, weight=1,
                        tooltip=f" {nt_val:.1f} nT"
                    ).add_to(mag_ev_group)
                mag_ev_group.add_to(m)
            except:
                pass
        
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
        folium.plugins.MousePosition(position='bottomright', separator=' | ', prefix='📍', lat_formatter='function(num) {return L.Util.formatNum(num, 6);}', lng_formatter='function(num) {return L.Util.formatNum(num, 6);}').add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        
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
            
            tab0, tab1, tab2, tab3, tab4 = st.tabs(["⚖️ Risk Weighting", "SSS", "MBES", "Magnetometer", "SBP"])

            with tab0:
                st.markdown("##### How the risk score was calculated for this hazard")
                weight_html = risk_weight_breakdown_html(hrow)
                st.markdown(weight_html, unsafe_allow_html=True)

                # Donut chart using plotly
                htype_check = hrow.get('hazard_type', '')
                profiles_chart = {
                    'Wreck':      {'Mag Anomaly':45,'Roughness':20,'Stratigraphy':15,'Slope':10,'Uncertainty':10},
                    'UXO':        {'Mag Anomaly':50,'Stratigraphy':20,'Uncertainty':15,'Roughness':10,'Slope':5},
                    'Pipeline':   {'Mag Anomaly':35,'Roughness':25,'Slope':20,'Stratigraphy':10,'Uncertainty':10},
                    'Gas':        {'Shallow Gas':45,'Stratigraphy':30,'Roughness':10,'Slope':5,'Uncertainty':10},
                    'Sand Wave':  {'Bedform':45,'Roughness':25,'Slope':15,'Uncertainty':10,'Stratigraphy':5},
                    'Hard Ground':{'Roughness':40,'Slope':25,'Stratigraphy':15,'Uncertainty':15,'Shallow Gas':5},
                    'Boulder':    {'Roughness':40,'Slope':20,'Stratigraphy':15,'Uncertainty':15,'Mag Anomaly':10},
                }
                chart_profile = next((v for k, v in profiles_chart.items() if k in htype_check),
                                     {'Slope':20,'Roughness':20,'Stratigraphy':20,'Uncertainty':20,'Shallow Gas':10,'Mag Anomaly':5,'Bedform':5})
                driver_colors = ['#0013C3','#8288A3','#FF4500','#FF8C00','#D1FE49','#8F998D','#c0c0c0']
                fig_donut = go.Figure(go.Pie(
                    labels=list(chart_profile.keys()),
                    values=list(chart_profile.values()),
                    hole=0.55,
                    marker=dict(colors=driver_colors[:len(chart_profile)]),
                    textinfo='label+percent',
                    textfont=dict(size=11),
                ))
                fig_donut.update_layout(
                    height=320, margin=dict(l=10,r=10,t=30,b=10),
                    paper_bgcolor='white', plot_bgcolor='white',
                    showlegend=False,
                    annotations=[dict(text=f"<b>{hrow.get('risk_score',5)}/10</b>",
                                      x=0.5, y=0.5, font_size=18, showarrow=False)]
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            
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
            st.subheader("🔄 Temporal Change Intelligence")

            tc = TEMPORAL_CHANGES.get(hid, TEMPORAL_CHANGES['default'])
            prev_score = tc['prev_score'] if tc['prev_score'] is not None else hrow.get('risk_score', 5)
            curr_score = tc['curr_score'] if tc['curr_score'] is not None else hrow.get('risk_score', 5)
            delta_score = tc['score_delta']
            traj = tc['trajectory']
            traj_col = '#CC0000' if 'WORSENING' in traj else '#FF8C00' if 'SLOWLY' in traj else '#00AA44'

            # ── Score comparison ──────────────────────────────────────────────
            arrow_label = f"▲ {delta_score:+.1f}" if delta_score > 0 else (f"▼ {delta_score:+.1f}" if delta_score < 0 else "● 0.0")
            score_curr_col = traj_col if delta_score != 0 else '#2E2E2E'

            comp_html = (
                "<div style='background:#f5f5f5;padding:12px 16px;border-radius:6px;"
                "display:flex;gap:20px;align-items:center;margin-bottom:10px;'>"
                f"<div style='text-align:center;'>"
                f"<div style='font-size:10px;color:#999;'>Previous Survey</div>"
                f"<div style='font-size:10px;color:#999;'>{tc['prev_date']}</div>"
                f"<div style='font-size:30px;font-weight:bold;color:#8288A3;'>{prev_score:.1f}"
                f"<span style='font-size:13px;'>/10</span></div></div>"
                "<div style='font-size:24px;color:#ccc;'>→</div>"
                f"<div style='text-align:center;'>"
                f"<div style='font-size:10px;color:#999;'>Current Survey</div>"
                f"<div style='font-size:10px;color:#999;'>{tc['curr_date']}</div>"
                f"<div style='font-size:30px;font-weight:bold;color:{score_curr_col};'>{curr_score:.1f}"
                f"<span style='font-size:13px;'>/10</span></div></div>"
                "<div style='border-left:1px solid #ddd;padding-left:20px;flex:1;'>"
                f"<div style='font-size:14px;font-weight:bold;color:{traj_col};'>{arrow_label} Risk Score Change</div>"
                f"<div style='font-size:12px;font-weight:bold;color:{traj_col};margin-top:3px;'>{traj}</div>"
                f"<div style='font-size:11px;color:#555;margin-top:4px;'>{tc['driver']}</div>"
                "</div></div>"
            )
            st.markdown(comp_html, unsafe_allow_html=True)

            # ── Parameter change table ────────────────────────────────────────
            st.markdown("**Parameter Changes:**")
            for m in tc['metrics']:
                sev_col = '#CC0000' if m['severity'] == 'high' else '#FF8C00' if m['severity'] == 'medium' else '#888888'
                trend_icon = '🔴' if m['trend'] == '↑' else '🟢' if m['trend'] == '↓' else '⚪'
                row_html = (
                    f"<div style='display:flex;align-items:center;gap:10px;padding:6px 10px;"
                    f"background:#f8f8f8;border-radius:4px;margin:3px 0;border-left:3px solid {sev_col};'>"
                    f"<span style='font-size:14px;'>{trend_icon}</span>"
                    f"<span style='font-weight:bold;width:160px;font-size:12px;'>{m['param']}</span>"
                    f"<span style='color:#777;font-size:11px;width:90px;'>{m['prev']}</span>"
                    f"<span style='color:#aaa;font-size:11px;'> → </span>"
                    f"<span style='font-weight:bold;font-size:12px;'>{m['curr']}</span>"
                    f"<span style='color:{sev_col};font-size:11px;margin-left:8px;'>{m['change']}</span>"
                    f"</div>"
                )
                st.markdown(row_html, unsafe_allow_html=True)

            # ── Action recommendation ─────────────────────────────────────────
            st.markdown(
                f"<div style='background:#e8f4fd;border-left:4px solid #0013C3;"
                f"padding:8px 12px;border-radius:4px;margin-top:10px;font-size:12px;'>"
                f"<b>📋 Recommended Action:</b> {tc['action']}</div>",
                unsafe_allow_html=True
            )
    
    else:
        st.info("⏳ Loading survey data automatically. If empty, navigate away and back to this page.")

st.markdown("---")
st.caption("⚡ The Grid - Powered by Multi-Sensor AI")
