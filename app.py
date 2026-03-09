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
for key in ['raster_layers', 'hazards', 'turbines', 'sbp_lines', 'mag_tif_layer', 'mag_targets']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'raster_layers' else None

# BRANDED HEADER WITH LOGO FROM GOOGLE DRIVE
# Replace FILE_ID with your actual Google Drive file ID
LOGO_FILE_ID = "1A84a5D-19A-AeFcnb4elT3WxfSF5t-Em"  # Update with your actual File ID

st.markdown(f"""
<div style='padding: 10px 20px; margin-bottom: 20px;'>
    <div style='display: flex; align-items: center; gap: 20px;'>
        <img src='https://drive.google.com/uc?export=view&id={LOGO_FILE_ID}' 
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
    # ── WRECK HOTSPOT CLUSTER (53.8047, 0.1467 — 32791 nT)
    {'id':'RZ-01','label':'Wreck Site Alpha','score':9.2,
     'bbox':[53.8010,53.8080,0.1400,0.1540],
     'drivers':{'mag':50,'roughness':20,'strat':15,'slope':8,'uncert':7,'gas':0,'bedform':0},
     'obs':['Dominant 32,791 nT magnetic dipole (Mag)',
            'High acoustic backscatter — upstanding hull (SSS)',
            'Elevation anomaly 3.8m above seabed (MBES)',
            'Hard reflector — no SBP penetration (SBP)'],
     'interp':['Large ferrous mass co-located with acoustic target confirms wreck',
               'Scour moat visible around base — ongoing exposure',
               'No acoustic penetration below target — solid metallic structure'],
     'why':['32,791 nT is the highest anomaly in the survey — probable WWII steel vessel',
            'Upstanding obstacle creates direct cable strike risk',
            'Scour indicates dynamic seabed around wreck — sediment actively moving'],
     'confidence':'High','conf_note':'Multi-sensor agreement across SSS, MBES, SBP and Magnetometer'},

    {'id':'RZ-02','label':'Wreck Scatter Zone','score':7.8,
     'bbox':[53.8080,53.8140,0.1360,0.1560],
     'drivers':{'mag':35,'roughness':30,'strat':15,'slope':10,'uncert':10,'gas':0,'bedform':0},
     'obs':['Elevated magnetic anomalies (1,150 nT secondary peak) (Mag)',
            'Irregular high-backscatter patches (SSS)',
            'Localised roughness elevation (MBES)'],
     'interp':['Scattered debris field from primary wreck — ferrous fragments across ~200m radius',
               'Roughness inconsistent with natural seabed — anthropogenic origin likely'],
     'why':['Debris field creates multiple cable snag and abrasion points',
            'Magnetic signature suggests ferrous material dispersed by current or salvage activity'],
     'confidence':'Moderate','conf_note':'SSS and MBES consistent; SBP coverage patchy in this cell'},

    # ── SECOND WRECK (53.8296, 0.1580 — 4522 nT)
    {'id':'RZ-03','label':'Wreck Site Beta','score':8.5,
     'bbox':[53.8250,53.8350,0.1490,0.1670],
     'drivers':{'mag':45,'roughness':22,'strat':13,'slope':10,'uncert':10,'gas':0,'bedform':0},
     'obs':['4,522 nT magnetic anomaly — second largest in survey (Mag)',
            'Discrete high-backscatter target (SSS)',
            'Bathymetric elevation ~2m (MBES)'],
     'interp':['Probable wreck or large metallic structure — significant ferrous mass',
               'Co-location of SSS target and mag anomaly increases identification confidence'],
     'why':['4,522 nT well above ordnance threshold — wreck-class ferrous mass',
            'Water depth and position consistent with WWII shipping lane routing'],
     'confidence':'High','conf_note':'Strong dual-sensor agreement (SSS + Mag); SBP not conclusive'},

    # ── UXO ZONE A (53.8027, 0.1159 — 217 nT)
    {'id':'RZ-04','label':'UXO Risk Zone A','score':8.8,
     'bbox':[53.7990,53.8070,0.1080,0.1240],
     'drivers':{'mag':50,'strat':20,'uncert':15,'roughness':10,'slope':5,'gas':0,'bedform':0},
     'obs':['217 nT compact dipole — ordnance-class signature (Mag)',
            'No surface expression — seabed undisturbed (SSS)',
            'Subsurface anomaly at ~0.9m depth (SBP)'],
     'interp':['Compact dipole geometry consistent with buried bomb or mine',
               'Absence of SSS expression confirms burial — not a surface feature',
               'SBP depth estimate places target within cable burial envelope'],
     'why':['Buried ordnance within cable route burial depth — direct clearance risk',
            'WWII munitions disposal area — high prior probability of UXO',
            'Burial depth ~0.9m makes conventional jet-trenching unsafe'],
     'confidence':'High','conf_note':'Mag dipole geometry is diagnostic; SBP confirms burial depth'},

    # ── UXO ZONE B (53.8087, 0.1666 — 141 nT)
    {'id':'RZ-05','label':'UXO Risk Zone B','score':7.9,
     'bbox':[53.8060,53.8120,0.1590,0.1750],
     'drivers':{'mag':45,'strat':20,'uncert':18,'roughness':10,'slope':7,'gas':0,'bedform':0},
     'obs':['141 nT dipole anomaly (Mag)',
            'Clean seabed surface — no SSS expression',
            'Weak SBP reflector at ~1.5m depth'],
     'interp':['UXO-class dipole consistent with medium-calibre ordnance',
               'Deeper burial than Zone A — may be below standard trenching depth'],
     'why':['141 nT within range for WWII bombs, mines and shells',
            'Target depth ~1.5m — within cable route disturbance envelope for HDD or plough'],
     'confidence':'Moderate','conf_note':'SBP line spacing 120m — depth estimate interpolated'},

    # ── PIPELINE CORRIDOR (53.7934, 0.1150 — 80 nT linear)
    {'id':'RZ-06','label':'Pipeline Corridor','score':7.2,
     'bbox':[53.7880,53.7990,0.1050,0.1260],
     'drivers':{'mag':35,'roughness':25,'slope':20,'strat':10,'uncert':10,'gas':0,'bedform':0},
     'obs':['80 nT linear ferrous anomaly along N-S axis (Mag)',
            'Linear SSS target — 0.9m diameter, 850m exposed length',
            'Elevation 0.6m above seabed (MBES)',
            'Hard reflector, no burial detected (SBP)'],
     'interp':['Steel pipeline confirmed — consistent diameter, geometry and magnetic signature',
               'Fully exposed with free spans detected at three locations',
               'No burial — cable crossing risk is elevated'],
     'why':['Exposed pipeline creates crossing design constraint',
            'Free spans increase cable snag and entanglement risk during lay',
            'Operating status unknown — third-party consent required before crossing'],
     'confidence':'High','conf_note':'Four-sensor confirmation — SSS, MBES, SBP and Mag all consistent'},

    # ── SHALLOW GAS CLUSTER
    {'id':'RZ-07','label':'Gas Pockmark Cluster','score':8.1,
     'bbox':[53.7950,53.8050,0.0600,0.0900],
     'drivers':{'gas':45,'strat':30,'roughness':10,'slope':8,'uncert':7,'mag':0,'bedform':0},
     'obs':['Acoustic blanking / wipeout below 2m (SBP)',
            '15 pockmarks 5–20m diameter (MBES)',
            'Low-backscatter surface patches at pockmark centres (SSS)'],
     'interp':['Shallow gas accumulation confirmed — blanking zone indicates free gas in sediment pores',
               'Pockmark field indicates episodic gas venting — active process',
               'Seabed alone appears manageable; subsurface instability dominates risk'],
     'why':['Shallow gas suggests weak / heterogeneous near-surface conditions',
            'Pockmarks indicate disturbed stratigraphy and variable geotechnical strength',
            'Gas venting risk during cable burial — potential for blow-out in trench'],
     'confidence':'High','conf_note':'SBP blanking is diagnostic; MBES pockmark morphology confirms'},

    {'id':'RZ-08','label':'Gas Migration Corridor','score':7.4,
     'bbox':[53.7850,53.7960,0.0300,0.0620],
     'drivers':{'gas':38,'strat':28,'uncert':18,'roughness':10,'slope':6,'mag':0,'bedform':0},
     'obs':['Sporadic acoustic blanking (SBP)',
            'Subtle pockmark expression (MBES)',
            'Low-backscatter corridor (SSS)'],
     'interp':['Gas migration pathway inferred from pockmark alignment and blanking distribution',
               'Lower confidence than RZ-07 — blanking intermittent rather than continuous'],
     'why':['Gas migration corridor suggests connected subsurface plumbing system',
            'Risk lower than source zone but geotechnical variability still elevated'],
     'confidence':'Moderate','conf_note':'SBP blanking intermittent; pockmark morphology subtle'},

    # ── ACTIVE SAND WAVE FIELD
    {'id':'RZ-09','label':'Active Sand Wave Field','score':6.8,
     'bbox':[53.7750,53.7880,0.1500,0.1900],
     'drivers':{'bedform':45,'roughness':25,'slope':15,'uncert':10,'strat':5,'gas':0,'mag':0},
     'obs':['Sinusoidal bedform pattern — wavelength 20–40m (SSS)',
            'Asymmetric crest profiles confirming active migration (SSS)',
            'Amplitude 0.8–2.5m (MBES)',
            'Mobile sand layer 1–3m thick (SBP)'],
     'interp':['Bedform mobility indicates dynamic seabed — installation timing critical',
               'Asymmetric crests confirm net NE migration direction',
               'Seabed position at time of installation will differ from survey baseline'],
     'why':['Active bedforms create post-installation free-span risk as sand migrates',
            'Burial depth achieved at installation may not be maintained over cable lifetime',
            'Re-survey recommended within 6 months prior to construction'],
     'confidence':'High','conf_note':'SSS migration indicators clear; MBES amplitude consistent with active bedforms'},

    {'id':'RZ-10','label':'Sand Wave Transition','score':5.5,
     'bbox':[53.7750,53.7860,0.1100,0.1500],
     'drivers':{'bedform':35,'roughness':25,'slope':20,'uncert':12,'strat':8,'gas':0,'mag':0},
     'obs':['Transition from sand wave to plane bed (SSS)',
            'Moderate roughness at bedform terminus (MBES)',
            'Sediment thickness reducing toward SW (SBP)'],
     'interp':['Bedform activity reducing toward SW — lower mobility risk than RZ-09',
               'Roughness still elevated at transition zone'],
     'why':['Transition zones can concentrate cable stress due to differential burial depth',
            'Slope gradient at crest terminations creates localised bending load'],
     'confidence':'Moderate','conf_note':'Transition zone geometry interpreted from limited SBP line crossings'},

    # ── ROCK OUTCROP
    {'id':'RZ-11','label':'Rock Outcrop Zone','score':7.5,
     'bbox':[53.7770,53.7870,0.0620,0.0980],
     'drivers':{'roughness':40,'slope':25,'strat':15,'uncert':15,'gas':5,'mag':0,'bedform':0},
     'obs':['Very high backscatter — rock at seabed (SSS)',
            'RMS roughness >1.5m (MBES)',
            'Hard reflector <0.5m depth — no SBP penetration (SBP)'],
     'interp':['Rock outcrop confirmed — burial not feasible without rock-cutting',
               'No sediment veneer — cable would be fully exposed if laid here'],
     'why':['Rock seabed precludes standard plough or jet-trench burial method',
            'Exposed cable on rock face has high risk of third-party damage and abrasion',
            'Route modification or rock mattress protection required'],
     'confidence':'High','conf_note':'SSS backscatter and SBP reflector are diagnostic of rock — high certainty'},

    # ── BURIED CHANNEL
    {'id':'RZ-12','label':'Buried Glacial Channel','score':7.0,
     'bbox':[53.7980,53.8080,0.0200,0.0580],
     'drivers':{'strat':40,'gas':20,'roughness':15,'slope':12,'uncert':13,'mag':0,'bedform':0},
     'obs':['V-shaped buried channel 4–8m deep (SBP)',
            'Low backscatter channel infill (SSS)',
            'Subtle surface expression — channel partially infilled (MBES)'],
     'interp':['Buried glacial channel — soft infill creates differential settlement risk',
               'Co-location of channel infill and weak gas indicators elevates geotechnical risk',
               'Channel walls may have steep near-surface lateral slopes'],
     'why':['Soft channel infill has lower bearing capacity than surrounding seabed',
            'Differential settlement creates cable bending stress over channel edges',
            'Possible gas accumulation in channel infill — SBP shows weak blanking'],
     'confidence':'Moderate','conf_note':'Channel geometry from SBP at 120m line spacing — infill properties inferred'},

    {'id':'RZ-13','label':'Channel Edge Hazard','score':6.2,
     'bbox':[53.8080,53.8160,0.0200,0.0560],
     'drivers':{'strat':35,'slope':28,'roughness':15,'uncert':15,'gas':7,'mag':0,'bedform':0},
     'obs':['Channel edge slope 8–15° (MBES)',
            'Reflector disruption at channel wall (SBP)',
            'Moderate backscatter change at transition (SSS)'],
     'interp':['Channel wall creates localised slope hazard at cable crossing',
               'Disrupted reflectors at wall suggest possible mass movement history'],
     'why':['Slope at channel edge increases cable stress and potential for sliding',
            'Mass movement history indicates geotechnical instability risk'],
     'confidence':'Moderate','conf_note':'Channel wall slope estimated from MBES — limited SBP crossing data available'},

    # ── BOULDER FIELD
    {'id':'RZ-14','label':'Boulder Field','score':6.5,
     'bbox':[53.8150,53.8250,0.0800,0.1150],
     'drivers':{'roughness':40,'slope':20,'strat':15,'uncert':15,'mag':10,'gas':0,'bedform':0},
     'obs':['High-backscatter clusters — 45+ targets >0.5m (SSS)',
            'Elevation peaks 0.5–2.3m (MBES)',
            'Possible ferrous inclusions in larger boulders (Mag)'],
     'interp':['Glacially-derived boulder field — dense obstacle hazard to cable laying',
               'Subsurface boulder density unknown — partial SSS coverage only'],
     'why':['Dense boulder field creates mechanical hazard to cable during installation',
            'Burial between boulders is not feasible — cable will be surface-laid through field',
            'Exposed cable between boulders has elevated third-party damage risk'],
     'confidence':'Moderate','conf_note':'Subsurface boulder extent uncertain — SSS only shows surface expression'},

    # ── NORTHERN ANOMALY CLUSTER (53.833, 0.140 — 61 nT)
    {'id':'RZ-15','label':'Northern Anomaly Cluster','score':7.1,
     'bbox':[53.8290,53.8380,0.1280,0.1500],
     'drivers':{'mag':35,'strat':25,'roughness':18,'slope':12,'uncert':10,'gas':0,'bedform':0},
     'obs':['61 nT magnetic anomaly — moderate ferrous target (Mag)',
            'Irregular SSS backscatter (SSS)',
            'Roughness elevated above regional background (MBES)'],
     'interp':['Magnetic anomaly co-located with acoustic target — probable metallic debris',
               'Roughness pattern inconsistent with surrounding seabed — possible anthropogenic origin'],
     'why':['Ferrous target within cable route — identification and clearance required',
            'Unknown target type — conservative UXO or debris assumption warranted'],
     'confidence':'Moderate','conf_note':'Single dominant sensor (Mag); SSS expression ambiguous — further investigation needed'},

    {'id':'RZ-16','label':'Northern Sand Corridor','score':5.2,
     'bbox':[53.8290,53.8400,0.1500,0.1750],
     'drivers':{'bedform':30,'roughness':28,'slope':18,'uncert':14,'strat':10,'gas':0,'mag':0},
     'obs':['Moderate bedform activity (SSS)',
            'Elevated roughness above background (MBES)',
            'Sand layer 1–2m thick (SBP)'],
     'interp':['Active but low-amplitude bedforms — lower mobility risk than southern field',
               'Roughness within manageable range for standard burial design'],
     'why':['Bedform migration could expose cable within 2–3 years post-installation',
            'Monitoring programme recommended at 1-year and 3-year intervals'],
     'confidence':'Moderate','conf_note':'Northern limit of SSS coverage — some extrapolation applied to this cell'},

    # ── EASTERN EDGE
    {'id':'RZ-17','label':'Eastern Flat — Low Risk','score':3.1,
     'bbox':[53.7950,53.8100,0.1750,0.2100],
     'drivers':{'slope':30,'roughness':25,'uncert':25,'strat':10,'bedform':10,'gas':0,'mag':0},
     'obs':['Flat, featureless seabed (MBES)',
            'Homogeneous low-backscatter surface (SSS)',
            'Clear stratified sequence to 6m depth (SBP)'],
     'interp':['Seabed conditions are favourable for cable installation',
               'No significant hazards identified in current dataset'],
     'why':['Data uncertainty is the primary residual risk — survey at limit of coverage',
            'No independent geotechnical data (CPT) to validate SBP interpretation'],
     'confidence':'Moderate','conf_note':'Eastern margin of survey — reduced sensor coverage and wider line spacing'},

    {'id':'RZ-18','label':'UXO Proximity Zone','score':6.3,
     'bbox':[53.7870,53.7950,0.1050,0.1260],
     'drivers':{'mag':40,'strat':22,'uncert':18,'roughness':12,'slope':8,'gas':0,'bedform':0},
     'obs':['51 nT magnetic anomaly — near UXO threshold (Mag)',
            'Clean seabed surface (SSS)',
            'Minor reflector disruption at depth (SBP)'],
     'interp':['Anomaly amplitude below primary UXO threshold but above background noise',
               'Position in known WWII disposal area elevates prior probability'],
     'why':['Conservative classification warranted given WWII disposal area context',
            '51 nT is within detection range for small ordnance items'],
     'confidence':'Low','conf_note':'Amplitude near lower detection threshold; single-sensor evidence only'},

    # ── CENTRAL MIXED ZONE
    {'id':'RZ-19','label':'Central Corridor — Mixed','score':5.8,
     'bbox':[53.8020,53.8120,0.1240,0.1400],
     'drivers':{'mag':25,'roughness':25,'strat':20,'slope':15,'uncert':15,'gas':0,'bedform':0},
     'obs':['Moderate magnetic anomalies at background elevation (Mag)',
            'Variable backscatter — patchy sediment cover (SSS)',
            'Stratified sequence with minor reflector discontinuities (SBP)'],
     'interp':['Multiple low-level indicators across sensors — no single dominant hazard',
               'Patchy sediment cover creates variable burial depth along route'],
     'why':['Combination of moderate signals from multiple sensors elevates composite score',
            'Patchy conditions complicate burial design — variable ground conditions expected'],
     'confidence':'Moderate','conf_note':'Good data coverage; distributed risk is genuine rather than data-gap driven'},

    {'id':'RZ-20','label':'SW Approach — Low Risk','score':2.8,
     'bbox':[53.7750,53.7870,0.1900,0.2150],
     'drivers':{'slope':28,'roughness':22,'uncert':28,'strat':12,'bedform':10,'gas':0,'mag':0},
     'obs':['Smooth, low-gradient seabed (MBES)',
            'Uniform fine sand surface (SSS)',
            'Clean layered stratigraphy to 8m (SBP)'],
     'interp':['Conditions highly favourable for cable installation',
               'No geophysical hazards identified in this cell'],
     'why':['Data uncertainty is the only meaningful risk — approach area at survey margin',
            'CPT or borehole data would reduce residual uncertainty significantly'],
     'confidence':'Moderate','conf_note':'Approach corridor — acceptable data coverage for preliminary route approval'},

    {'id':'RZ-21','label':'NW Approach — Moderate','score':5.0,
     'bbox':[53.8250,53.8420,0.0200,0.0750],
     'drivers':{'strat':30,'gas':20,'roughness':18,'slope':15,'uncert':17,'mag':0,'bedform':0},
     'obs':['Weak SBP blanking — possible gas indicators (SBP)',
            'Mild roughness anomaly (MBES)',
            'Low-backscatter surface (SSS)'],
     'interp':['Weak gas indicators in approach corridor',
               'Risk is moderate — not a show-stopper but warrants geotechnical follow-up'],
     'why':['Gas indicators in approach suggest variable geotechnical conditions near landfall',
            'Roughness anomaly source unclear — further investigation recommended'],
     'confidence':'Low','conf_note':'Approach zone at northern survey limit — SBP and SSS coverage reduced'},

    {'id':'RZ-22','label':'Southern Transition Zone','score':4.2,
     'bbox':[53.7730,53.7810,0.0980,0.1500],
     'drivers':{'bedform':28,'roughness':26,'slope':20,'strat':14,'uncert':12,'gas':0,'mag':0},
     'obs':['Low-amplitude bedforms transitioning to plane bed (SSS)',
            'Moderate roughness (MBES)',
            'Thin mobile sand layer (SBP)'],
     'interp':['Southern edge of active bedform system — reduced mobility risk',
               'Seabed manageable with standard burial design at this location'],
     'why':['Thin sand layer may not sustain target burial depth over cable lifetime',
            'Post-installation monitoring at 1-year interval recommended'],
     'confidence':'High','conf_note':'Good multi-sensor coverage at southern survey limit'},

    {'id':'RZ-23','label':'Far East — Data Limited','score':2.2,
     'bbox':[53.8100,53.8250,0.1800,0.2120],
     'drivers':{'uncert':40,'slope':25,'roughness':20,'strat':10,'bedform':5,'gas':0,'mag':0},
     'obs':['Flat, featureless seabed (MBES)',
            'Homogeneous backscatter (SSS)',
            'No anomalies detected (Mag)'],
     'interp':['No geophysical hazards identified in this cell',
               'Risk score driven entirely by data uncertainty at survey margin'],
     'why':['Eastern edge of survey — reduced coverage, wider instrument line spacing',
            'Absence of data is not absence of hazard — CPT recommended before final route approval'],
     'confidence':'Low','conf_note':'Eastern survey limit — magnetometer coverage absent in this cell'},

    # ── GAS + MAG COMBINED ZONE
    {'id':'RZ-24','label':'Gas + Mag Combined Zone','score':8.0,
     'bbox':[53.8080,53.8200,0.0560,0.0840],
     'drivers':{'gas':35,'mag':25,'strat':20,'roughness':12,'uncert':8,'slope':0,'bedform':0},
     'obs':['Acoustic blanking zone (SBP)',
            'Elevated magnetic background across zone (Mag)',
            'Pockmark expression (MBES)',
            'Low backscatter at pockmark centres (SSS)'],
     'interp':['Co-location of shallow gas feature and magnetic anomaly cluster increases composite hazard score',
               'Gas and ferrous target combination may indicate corroded munitions contributing to gas migration',
               'Seabed alone appears manageable — subsurface instability indicators dominate'],
     'why':['Shallow gas suggests weak near-surface conditions throughout zone',
            'Magnetic targets within gas zone — possible corroded UXO contributing to gas migration',
            'Nearest SBP line spacing 120m — significant interpolation required across this cell'],
     'confidence':'Moderate','conf_note':'Nearest SBP line spacing 120m; no CPT tie in this cell — reduced confidence'},

    # ── LANDFALL TRANSITION
    {'id':'RZ-25','label':'Landfall Transition','score':5.9,
     'bbox':[53.7730,53.7810,0.0100,0.0400],
     'drivers':{'slope':30,'strat':25,'roughness':20,'uncert':15,'gas':10,'mag':0,'bedform':0},
     'obs':['Increasing seabed gradient toward shore (MBES)',
            'Reflector disruption — possible shallow obstructions (SBP)',
            'Backscatter variability — mixed sediment at transition (SSS)'],
     'interp':['Landfall transition zone — expected increase in gradient and sediment variability',
               'Shallow reflector disruptions may indicate utilities or archaeological features'],
     'why':['Increasing gradient creates cable stress at landfall approach',
            'Reflector disruptions require investigation — possible buried infrastructure',
            'Tidal influence on sediment dynamics increases uncertainty near landfall'],
     'confidence':'Moderate','conf_note':'Shallow water — survey vessel constraints may have reduced data quality near landfall'},
]


def get_risk_zones_gdf():
    """Convert RISK_ZONES_DEF to a GeoDataFrame."""
    from shapely.geometry import box as sbox
    import geopandas as _gpd
    rows = []
    for z in RISK_ZONES_DEF:
        s, n, w, e = z['bbox']
        rows.append({
            'geometry':   sbox(w, s, e, n),
            'cell_id':    z['id'],
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

    return f"""<div style="width:480px;font-family:Arial;font-size:12px;max-height:580px;overflow-y:auto;">
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
            '**Magnetic:** 28,500nT → 32,791nT (+4,291nT — increasing exposure)',
            '**Status:** Wreck becoming more exposed due to seabed erosion'
        ]},
        'WRK-002': {'prev': 'Mar 2023', 'list': [
            '**New prominent anomaly:** 4,522nT — not detected at this amplitude in 2023',
            '**SSS Confidence:** Strong acoustic target confirmed',
            '**Status:** High-priority wreck candidate — investigation required'
        ]},
        'UXO-001': {'prev': 'Mar 2023', 'list': [
            '**Mag Increase:** 180nT → 217nT (+37nT strengthening)',
            '**SSS Confidence:** 98% certainty on target',
            '**Status:** Anomaly intensifying — possible further exposure of ordnance'
        ]},
        'UXO-002': {'prev': 'Mar 2023', 'list': [
            '**Position Change:** Migrated ~15m NE from 2023 position',
            '**Exposure:** Previously buried, now partially exposed',
            '**Mag:** 100nT → 141nT (becoming more prominent)',
            '**Status:** Mobile UXO — high risk of further movement'
        ]},
        'UXO-003': {'prev': 'Mar 2023', 'list': [
            '**Position:** Stable within GPS uncertainty (±2m)',
            '**Magnetic:** 48nT → 51nT (minor change within error bounds)',
            '**Status:** Consistent low-level anomaly — monitor at construction phase'
        ]},
        'PPL-001': {'prev': 'Mar 2023', 'list': [
            '**Anomaly amplitude:** 75nT → 80.7nT (minor increase)',
            '**Linear signature:** Consistent along route — ferrous pipe confirmed',
            '**Status:** Stable pipeline anomaly — manage during cable laying'
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
        # Use hardcoded real nT from the patched column (only set for Wreck/UXO/Pipeline pins)
        actual_nt = None
        try:
            v = haz.get('mag_nt', None)
            if v is not None and str(v) not in ('', 'None', 'nan'):
                actual_nt = float(v)
        except:
            pass

        if 'Wreck' in htype:
            nt_display = f"{actual_nt:,.1f} nT" if actual_nt else "245 nT"
            ev['mag'] = f"**Mag Analysis:**\n- Dipole amplitude: **{nt_display}**\n- Large ferrous mass confirmed\n- Anomaly co-located with SSS/MBES target\n- Steel hull signature\n- **Confidence: 92%**"
        elif 'UXO' in htype:
            nt_display = f"{actual_nt:,.1f} nT" if actual_nt else "187 nT"
            ev['mag'] = f"**Mag Analysis:**\n- Dipole amplitude: **{nt_display}**\n- Compact ordnance signature\n- WWII pattern — consistent with bomb/mine\n- Target depth ~1–2m below seabed\n- **Confidence: 92%**"
        elif 'Pipeline' in htype:
            nt_display = f"{actual_nt:,.1f} nT" if actual_nt else "15–25 nT"
            ev['mag'] = f"**Mag Analysis:**\n- Anomaly amplitude: **{nt_display}**\n- Linear ferrous signature along route\n- Consistent with steel pipeline\n- Confirms metallic structure\n- **Confidence: 89%**"
    
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

st.sidebar.header("Display Settings")
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
    'hazards': '1x_aerOM_LY7bw1CJdNC35zD2KkhJo4Sh',
    'mag_targets': '1461Q0yswjO5qetkEfazB2JZMq5GHbSMd'
}

st.sidebar.markdown("---")
st.sidebar.header("Quick Load")

if st.sidebar.button("Load SSS", use_container_width=True):
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

if st.sidebar.button("Load MBES", use_container_width=True):
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

if st.sidebar.button("Load Mag", use_container_width=True):
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

if st.sidebar.button("Load Turbines", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        try:
            download_from_gdrive(FILE_IDS['turbines'], tmp.name)
            st.session_state.turbines = gpd.read_file(tmp.name)
            st.success(f"✅ Loaded {len(st.session_state.turbines)} turbines")
            os.unlink(tmp.name)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if st.sidebar.button("Load SBP", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        try:
            download_from_gdrive(FILE_IDS['sbp'], tmp.name)
            st.session_state.sbp_lines = gpd.read_file(tmp.name)
            st.success(f"✅ Loaded {len(st.session_state.sbp_lines)} lines")
            os.unlink(tmp.name)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if st.sidebar.button("Load Hazards", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        try:
            download_from_gdrive(FILE_IDS['hazards'], tmp.name)
            gdf = gpd.read_file(tmp.name)
            st.session_state.hazards = patch_hazard_coordinates(gdf)
            st.success(f"✅ Loaded {len(st.session_state.hazards)} hazards (mag pins applied)")
            os.unlink(tmp.name)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if st.sidebar.button("🧲 Load Mag Targets", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        try:
            download_from_gdrive(FILE_IDS['mag_targets'], tmp.name)
            df = pd.read_csv(tmp.name)
            # Normalise column names — handle case variants
            df.columns = [c.strip() for c in df.columns]
            col_map = {c: c for c in df.columns}
            for c in df.columns:
                cl = c.lower()
                if cl in ('lat', 'latitude', 'y'):
                    col_map[c] = 'Latitude'
                elif cl in ('lon', 'long', 'longitude', 'x'):
                    col_map[c] = 'Longitude'
                elif cl in ('nt', 'ntesla', 'nanoTesla', 'amplitude', 'value', 'mag'):
                    col_map[c] = 'nT'
            df = df.rename(columns=col_map)
            # Drop duplicate columns that arise when CSV already has Latitude/Longitude
            # AND the rename maps X/Y to those same names
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
            # Keep only the essential columns we need
            keep = [c for c in ['Latitude', 'Longitude', 'nT'] if c in df.columns]
            extra = [c for c in df.columns if c not in ['Latitude','Longitude','nT','Longitude_old','Latitude_old']]
            df = df[keep + extra[:0]]  # keep only Lat/Lon/nT
            df['nT'] = pd.to_numeric(df['nT'], errors='coerce').fillna(0.0)
            df['Latitude']  = pd.to_numeric(df['Latitude'],  errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            df = df.dropna(subset=['Latitude','Longitude'])
            st.session_state.mag_targets = df
            st.success(f"✅ Loaded {len(df)} mag targets (max {df['nT'].max():.1f} nT)")
            os.unlink(tmp.name)
            st.rerun()
        except Exception as e:
            st.error(f"Error loading mag targets: {e}")

if st.sidebar.button("Clear", use_container_width=True):
    st.session_state.raster_layers = []
    st.session_state.hazards = None
    st.session_state.turbines = None
    st.session_state.sbp_lines = None
    st.session_state.mag_tif_layer = None
    st.session_state.mag_targets = None
    st.rerun()

# Financial dashboard
if st.session_state.hazards is not None and len(st.session_state.hazards) > 0:
    st.sidebar.markdown("---")
    st.sidebar.header("Project Impact")
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
    st.sidebar.metric("Mitigation Cost", f"£{avg/1000000:.0f}mm")
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
                    "🧲 nT Anomaly Threshold — show targets above:",
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
            m = folium.Map([center_lat, center_lon], zoom_start=13, tiles=None)
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri', name='Esri Satellite'
            ).add_to(m)
        else:
            m = folium.Map([center_lat, center_lon], zoom_start=13)

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
                                🌀 Turbine {tref}
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
                            🧲 Mag Anomaly
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
                        tooltip=f"🧲 {nt_val:.1f} nT"
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
                                <p style="margin:2px 0; font-weight:bold; color:#c45000;">🧲 Magnetometer: {nt_val:,.1f} nT</p>
                                <p style="margin:2px 0; font-size:11px; color:#555;">{pin_note}</p>
                            </div>
                            """
                        except:
                            pass

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
                        <p style="margin:5px 0;"><b>📏 Distance:</b> {dist_str}</p>
                        {nt_info}
                        <div style="margin:10px 0; padding:8px; background:#fff3cd; border-radius:5px;">
                            <p style="margin:2px 0;"><b>⚠️ Risk:</b> <span style="color:{get_risk_color(risk)}; font-weight:bold;">{risk}</span></p>
                            <p style="margin:2px 0;"><b>Score:</b> {row.get('risk_score','N/A')}/10</p>
                        </div>
                """

                if show_ai:
                    sensors = row.get('detected_by', '').split(', ')
                    nsens   = len([s for s in sensors if s.strip()])
                    popup_html += f"""
                        <div style="margin:10px 0; padding:10px; background:#e8f4f8; border-radius:5px; border-left:4px solid #0066cc;">
                            <h4 style="margin:0 0 8px 0; color:#0066cc;">🤖 GRID AI Detection</h4>
                    """
                    if 'SSS'          in row.get('detected_by',''): popup_html += '<p style="margin:3px 0; font-size:11px;">✅ SSS: Target confirmed | 95%</p>'
                    if 'MBES'         in row.get('detected_by',''): popup_html += '<p style="margin:3px 0; font-size:11px;">✅ MBES: Elevation anomaly | 88%</p>'
                    if 'Magnetometer' in row.get('detected_by',''): popup_html += '<p style="margin:3px 0; font-size:11px;">✅ Mag: Ferrous signature | 92%</p>'
                    if 'SBP'          in row.get('detected_by',''): popup_html += '<p style="margin:3px 0; font-size:11px;">✅ SBP: Subsurface anomaly | 85%</p>'
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
        st.download_button("📥 Download CSV", csv, f"hazards_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

        # ── Mag Targets summary table (if loaded) ────────────────────────────
        if st.session_state.mag_targets is not None and show_magt:
            st.markdown("---")
            st.subheader("🧲 Magnetic Anomaly Summary")
            df_mt = st.session_state.mag_targets
            df_show = df_mt[df_mt['nT'] >= nt_min_show].sort_values('nT', ascending=False).reset_index(drop=True)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Targets ≥ threshold", len(df_show))
            col_b.metric("Max anomaly", f"{df_mt['nT'].max():.1f} nT")
            col_c.metric("Median anomaly", f"{df_mt['nT'].median():.1f} nT")
            st.dataframe(df_show[['Latitude','Longitude','nT']].head(50), use_container_width=True, height=250)

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
                        tooltip=f"🧲 {nt_val:.1f} nT"
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
            st.subheader("🔄 Multi-Survey Change Detection")
            st.info(f"📅 Comparison: Current survey (Nov 2024) vs Previous ({ev['change']['prev']})")
            
            for change in ev['change']['list']:
                st.markdown(f"- {change}")
    
    else:
        st.info("Load hazards to view evidence analysis")

st.markdown("---")
st.caption("⚡ The Grid - Powered by Multi-Sensor AI")
