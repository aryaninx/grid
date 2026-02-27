# Marine Survey Viewer

Interactive web application for visualizing marine survey data including MBES bathymetry, sidescan sonar imagery, and sub-bottom profiler tracklines.

## Features

- 🗺️ Interactive OpenStreetMap display
- 📊 MBES bathymetry visualization
- 📡 Sidescan sonar imagery overlay
- 📏 Survey trackline display
- ☁️ Cloud data loading from Google Drive
- 🎨 Multiple basemap and colormap options

## Live Demo

🔗 [View Live App](https://your-app-name.streamlit.app)

## Data Sources

The app supports three data loading methods:

1. **Sample Data** - Included in repository (`data/sample/`)
2. **File Upload** - Upload directly from your computer
3. **Google Drive** - Load large files from cloud storage

## Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/grid-demo.git
cd grid-demo

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## Deployment

This app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) (free hosting).

To deploy your own:
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Click Deploy!

## Data Format

- **MBES**: GeoTIFF format, any CRS (auto-converted to WGS84)
- **SSS**: GeoTIFF format (RGB or grayscale)
- **Vectors**: GeoJSON format (tracklines, footprints, infrastructure)

## Usage

### Using Sample Data
1. Select "Sample Data (GitHub)" in sidebar
2. Map loads automatically

### Using Google Drive
1. Upload files to Google Drive
2. Set sharing to "Anyone with link can view"
3. Copy File ID from share link
4. Paste into app sidebar
5. Click Load

### Getting Google Drive File ID

From this link:
```
https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view
```

The File ID is: `1a2b3c4d5e6f7g8h9i0j`

## License

MIT License - Feel free to use for your projects!

## Contact

Built for The Grid - Marine survey data platform
