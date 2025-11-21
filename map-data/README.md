Place offline MBTiles datasets here.

You do not need these files for basic usage.
The app can run with free online map services by default.

Required for Offline Map Mode:
- satellite.mbtiles

Optional for 3D terrain rendering:
- terrain-rgb.mbtiles

Notes:
- Keep filenames exactly as above.
- These files are intentionally git-ignored because they are large.
- Restart backend after adding or replacing files.
- Set `MAP_ONLINE_FALLBACK=false` only if you want to force offline mode.

Suggested data sources:
- Satellite imagery:
	- MapTiler Satellite MBTiles exports (commercial, easiest for offline global use)
	- Sentinel-2 derived global mosaics (open data; requires building your own MBTiles)
- Terrain (DEM / terrain-rgb):
	- Copernicus DEM GLO-30 (global 30m DEM)
	- NASADEM / SRTM (global near-30m DEM)

Important:
- A true whole-world HD satellite dataset is extremely large (hundreds of GB to multi-TB).
- Do not commit these files to git; store them locally and mount them with Docker.
