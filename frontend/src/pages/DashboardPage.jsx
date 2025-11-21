import { useEffect, useMemo, useState } from 'react'
import BeforeAfterSlider from '../components/BeforeAfterSlider'
import MapAreaPicker3D from '../components/MapAreaPicker3D'
import UploadDropzone from '../components/UploadDropzone'
import { API_BASE_URL, makeApiUrl } from '../config'

const FALLBACK_CLASSES = [
  { index: 0, name: 'Class 0', color: '#ff0000' },
  { index: 1, name: 'Class 1', color: '#00ff00' },
  { index: 2, name: 'Class 2', color: '#0000ff' },
  { index: 3, name: 'Class 3', color: '#ffff00' },
]

const MIN_AREA_ZOOM = 10
const MAX_SOURCE_TILES = 144
const DEFAULT_TILE_OVERLAP = 48
const DEFAULT_CONFIDENCE_THRESHOLD = 0.5
const DEFAULT_BACKGROUND_CLASS = 0


function clipLat(lat) {
  return Math.max(-85.05112878, Math.min(85.05112878, lat))
}


function lonLatToGlobalPixel(lon, lat, zoom) {
  const clampedLat = clipLat(lat)
  const worldSize = (2 ** zoom) * 256
  const px = ((lon + 180) / 360) * worldSize
  const sinLat = Math.sin((clampedLat * Math.PI) / 180)
  const py = (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * worldSize
  return { px, py }
}


function validateAreaSelectionQuality(bounds, zoom) {
  if (zoom < MIN_AREA_ZOOM) {
    return `Please zoom in to at least level ${MIN_AREA_ZOOM} before running segmentation.`
  }

  const topLeft = lonLatToGlobalPixel(bounds.west, bounds.north, zoom)
  const bottomRight = lonLatToGlobalPixel(bounds.east, bounds.south, zoom)
  const widthPx = Math.max(1, bottomRight.px - topLeft.px)
  const heightPx = Math.max(1, bottomRight.py - topLeft.py)

  const tilesW = Math.ceil(widthPx / 256)
  const tilesH = Math.ceil(heightPx / 256)
  const tileCount = tilesW * tilesH
  if (tileCount > MAX_SOURCE_TILES) {
    return 'Selected area is too large. Please zoom in and select a smaller region.'
  }

  return null
}

function DashboardPage() {
  const [inputMode, setInputMode] = useState('upload')
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [selectedBounds, setSelectedBounds] = useState(null)

  const [useTiling, setUseTiling] = useState(true)
  const [tileSize, setTileSize] = useState(384)
  const [batchSize, setBatchSize] = useState(8)
  const [overlayAlpha, setOverlayAlpha] = useState(0.45)
  const [mapOpacity, setMapOpacity] = useState(0.65)
  const [activeLayer, setActiveLayer] = useState('prediction')

  const classes = useMemo(() => {
    if (!result?.classes) {
      return FALLBACK_CLASSES
    }

    return result.classes.map((item) => {
      const fallback = FALLBACK_CLASSES.find((entry) => entry.index === item.index)
      return {
        index: item.index,
        name: item.name,
        color: fallback?.color || '#999999',
      }
    })
  }, [result])

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  const onFileSelected = (selectedFile) => {
    setError('')
    setResult(null)

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setFile(selectedFile)
    setPreviewUrl(URL.createObjectURL(selectedFile))
  }

  const runPrediction = async () => {
    setIsLoading(true)
    setError('')
    let timeoutId = null

    try {
      let response
      const controller = new AbortController()
      const timeoutMs = inputMode === 'map' ? 120000 : 90000
      timeoutId = window.setTimeout(() => controller.abort(), timeoutMs)

      if (inputMode === 'upload') {
        if (!file) {
          throw new Error('Upload an image first.')
        }

        const formData = new FormData()
        formData.append('file', file)
        formData.append('use_tiling', String(useTiling))
        formData.append('tile_size', String(tileSize))
        formData.append('batch_size', String(batchSize))
        formData.append('tile_overlap', String(DEFAULT_TILE_OVERLAP))
        formData.append('confidence_threshold', String(DEFAULT_CONFIDENCE_THRESHOLD))
        formData.append('background_class', String(DEFAULT_BACKGROUND_CLASS))
        formData.append('overlay_alpha', String(overlayAlpha))
        formData.append('export_geotiff', 'true')

        response = await fetch(makeApiUrl('/predict'), {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        })
      } else {
        if (!selectedBounds) {
          throw new Error('Select and confirm an area on the map first.')
        }

        const normalizedBounds = {
          south: Math.min(selectedBounds.south, selectedBounds.north),
          north: Math.max(selectedBounds.south, selectedBounds.north),
          west: Math.min(selectedBounds.west, selectedBounds.east),
          east: Math.max(selectedBounds.west, selectedBounds.east),
        }
        const requestedZoom = Math.max(
          0,
          Math.min(17, Math.ceil(selectedBounds?.map_zoom ?? 13)),
        )
        const qualityValidationError = validateAreaSelectionQuality(normalizedBounds, requestedZoom)
        if (qualityValidationError) {
          throw new Error(qualityValidationError)
        }

        response = await fetch(makeApiUrl('/predict/area'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...normalizedBounds,
            zoom: requestedZoom,
            use_tiling: useTiling,
            tile_size: tileSize,
            batch_size: batchSize,
            tile_overlap: DEFAULT_TILE_OVERLAP,
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            background_class: DEFAULT_BACKGROUND_CLASS,
            overlay_alpha: overlayAlpha,
          }),
          signal: controller.signal,
        })
      }

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed.')
      }

      setResult(data)
    } catch (requestError) {
      if (requestError?.name === 'AbortError') {
        setError('Request timed out. Try a smaller area or lower tile size/batch size.')
      } else {
        setError(requestError.message || 'Unexpected API error.')
      }
      setResult(null)
    } finally {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId)
      }
      setIsLoading(false)
    }
  }

  const getDownloadUrl = (path) => {
    if (!path) {
      return null
    }
    return makeApiUrl(path)
  }

  const mapPredictionSrc =
    getDownloadUrl(result?.downloads?.prediction_png) || result?.images_base64?.prediction || ''
  const mapOriginalSrc =
    getDownloadUrl(result?.downloads?.original_png) || result?.images_base64?.original || ''
  const sliderOriginalSrc =
    getDownloadUrl(result?.downloads?.original_png) || result?.images_base64?.original || previewUrl
  const sliderPredictionSrc =
    getDownloadUrl(result?.downloads?.prediction_png) || result?.images_base64?.prediction || ''
  const needsMoreZoom = inputMode === 'map' && selectedBounds?.map_zoom && selectedBounds.map_zoom < MIN_AREA_ZOOM

  return (
    <main className="app-shell">
      <header className="hero-header">
        <p className="eyebrow">AI Geospatial Intelligence</p>
        <h1>Satellite Land-Use Segmentation Studio</h1>
        <p>
          Explore an offline satellite world map, select an area of interest, run semantic
          segmentation, and inspect overlays in 3D.
        </p>
        <span className="api-tag">Backend: {API_BASE_URL}</span>
      </header>

      <section className="workspace-grid">
        {inputMode === 'upload' ? (
          <UploadDropzone
            file={file}
            previewUrl={previewUrl}
            onFileSelected={onFileSelected}
            disabled={isLoading}
          />
        ) : (
          <section className="panel upload-panel map-mode-panel">
            <div className="panel-head">
              <h2>Map Area Input</h2>
              <p>
                No image upload is needed. Use the map panel below: start selection, click two corners,
                confirm the area, then run segmentation.
              </p>
            </div>
            <div className="preview-empty map-mode-box">
              <p>Selection status: {selectedBounds ? 'Area selected' : 'No area selected yet'}</p>
            </div>
          </section>
        )}

        <section className="panel controls-panel">
          <div className="panel-head">
            <h2>Inference Controls</h2>
            <p>Adjust speed, quality, and map display behavior.</p>
          </div>

          <div className="toggle-row">
            <button
              className={`toggle-btn ${inputMode === 'upload' ? 'is-active' : ''}`}
              onClick={() => setInputMode('upload')}
            >
              Upload Image
            </button>
            <button
              className={`toggle-btn ${inputMode === 'map' ? 'is-active' : ''}`}
              onClick={() => setInputMode('map')}
            >
              Select Map Area
            </button>
          </div>

          <p className="map-mode-hint">
            {inputMode === 'upload'
              ? 'Upload flow: choose a local image, then run segmentation.'
              : 'Map flow: Start selection, pick two corners, confirm, then run segmentation.'}
          </p>

          {needsMoreZoom ? (
            <p className="error-text">
              Zoom in to at least level {MIN_AREA_ZOOM} before running segmentation for better quality.
            </p>
          ) : null}

          <label>
            <span>Use Tiled Inference</span>
            <input
              type="checkbox"
              checked={useTiling}
              onChange={(event) => setUseTiling(event.target.checked)}
            />
          </label>

          <label>
            <span>Tile Size</span>
            <input
              type="number"
              min={64}
              max={1024}
              step={32}
              value={tileSize}
              onChange={(event) => setTileSize(Number(event.target.value) || 384)}
              disabled={!useTiling}
            />
          </label>

          <label>
            <span>Batch Size</span>
            <input
              type="number"
              min={1}
              max={32}
              value={batchSize}
              onChange={(event) => setBatchSize(Number(event.target.value) || 8)}
            />
          </label>

          <label>
            <span>Overlay Alpha ({overlayAlpha.toFixed(2)})</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={overlayAlpha}
              onChange={(event) => setOverlayAlpha(Number(event.target.value))}
            />
          </label>

          <button
            onClick={runPrediction}
            className="primary-btn"
            disabled={isLoading || (inputMode === 'upload' ? !file : !selectedBounds)}
          >
            {isLoading
              ? 'Running segmentation...'
              : inputMode === 'upload'
                ? 'Run Segmentation on Uploaded Image'
                : 'Run Segmentation on Selected Area'}
          </button>

          {error ? <p className="error-text">{error}</p> : null}

          {result ? (
            <div className="run-stats">
              <p>
                Inference: <strong>{result.timing?.inference_ms} ms</strong>
              </p>
              <p>
                Size: <strong>{result.shape?.width} x {result.shape?.height}</strong>
              </p>
              {inputMode === 'map' && selectedBounds ? (
                <p>
                  Area: <strong>{selectedBounds.south.toFixed(4)}, {selectedBounds.west.toFixed(4)} to {selectedBounds.north.toFixed(4)}, {selectedBounds.east.toFixed(4)}</strong>
                </p>
              ) : null}
              {inputMode === 'map' && selectedBounds?.map_zoom ? (
                <p>
                  Selected Zoom: <strong>{selectedBounds.map_zoom.toFixed(2)}</strong>
                </p>
              ) : null}
            </div>
          ) : null}
        </section>

        <section className="panel legend-panel">
          <div className="panel-head">
            <h2>Land-Use Classes</h2>
            <p>Prediction color mapping used by model output.</p>
          </div>
          <ul className="legend-list">
            {classes.map((entry) => (
              <li key={entry.index}>
                <span className="legend-swatch" style={{ backgroundColor: entry.color }} />
                <span>{entry.index}</span>
                <span>{entry.name}</span>
              </li>
            ))}
          </ul>
        </section>
      </section>

      <section className="workspace-grid lower-grid">
        <BeforeAfterSlider
          originalSrc={sliderOriginalSrc}
          predictionSrc={sliderPredictionSrc}
        />

        <section className="panel controls-panel map-controls-panel">
          <div className="panel-head">
            <h2>Map Controls</h2>
            <p>Control layer visibility and export your results.</p>
          </div>

          {inputMode === 'map' ? (
            <>
              <label>
                <span>Overlay Opacity ({mapOpacity.toFixed(2)})</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={mapOpacity}
                  onChange={(event) => setMapOpacity(Number(event.target.value))}
                />
              </label>

              <div className="toggle-row">
                <button
                  className={`toggle-btn ${activeLayer === 'prediction' ? 'is-active' : ''}`}
                  onClick={() => setActiveLayer('prediction')}
                >
                  Prediction Layer
                </button>
                <button
                  className={`toggle-btn ${activeLayer === 'original' ? 'is-active' : ''}`}
                  onClick={() => setActiveLayer('original')}
                >
                  Original Layer
                </button>
              </div>
            </>
          ) : null}

          <a
            className={`secondary-btn ${result?.downloads?.prediction_png ? '' : 'is-disabled'}`}
            href={getDownloadUrl(result?.downloads?.prediction_png) || '#'}
            download
          >
            Download PNG
          </a>

          <a
            className={`secondary-btn ${result?.downloads?.mask_geotiff ? '' : 'is-disabled'}`}
            href={getDownloadUrl(result?.downloads?.mask_geotiff) || '#'}
            download
          >
            Download GeoTIFF
          </a>
        </section>
      </section>

      {inputMode === 'map' ? (
        <MapAreaPicker3D
          onSelectionChange={setSelectedBounds}
          overlayImageUrl={activeLayer === 'original' ? mapOriginalSrc : mapPredictionSrc}
          overlayBounds={result?.geo?.bounds_epsg4326}
          overlayOpacity={mapOpacity}
        />
      ) : null}
    </main>
  )
}

export default DashboardPage
