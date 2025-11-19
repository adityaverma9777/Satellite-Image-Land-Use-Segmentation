import { useMemo, useState } from 'react'
import BeforeAfterSlider from '../components/BeforeAfterSlider'
import MapOverlayView from '../components/MapOverlayView'
import UploadDropzone from '../components/UploadDropzone'
import { API_BASE_URL, makeApiUrl } from '../config'

const FALLBACK_CLASSES = [
  { index: 0, name: 'Urban', color: '#ff0000' },
  { index: 1, name: 'Agriculture', color: '#ffff00' },
  { index: 2, name: 'Rangeland', color: '#a52a2a' },
  { index: 3, name: 'Forest', color: '#228b22' },
  { index: 4, name: 'Water', color: '#0000ff' },
  { index: 5, name: 'Barren', color: '#808080' },
]

function DashboardPage() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const [useTiling, setUseTiling] = useState(true)
  const [tileSize, setTileSize] = useState(256)
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

  const onFileSelected = (selectedFile) => {
    setError('')
    setResult(null)
    setFile(selectedFile)

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setPreviewUrl(URL.createObjectURL(selectedFile))
  }

  const runPrediction = async () => {
    if (!file) {
      setError('Please upload a satellite image first.')
      return
    }

    setIsLoading(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('use_tiling', String(useTiling))
      formData.append('tile_size', String(tileSize))
      formData.append('batch_size', String(batchSize))
      formData.append('overlay_alpha', String(overlayAlpha))
      formData.append('export_geotiff', 'true')

      const response = await fetch(makeApiUrl('/predict'), {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed.')
      }

      setResult(data)
    } catch (requestError) {
      setError(requestError.message || 'Unexpected API error.')
      setResult(null)
    } finally {
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

  return (
    <main className="app-shell">
      <header className="hero-header">
        <p className="eyebrow">AI Geospatial Intelligence</p>
        <h1>Satellite Land-Use Segmentation Studio</h1>
        <p>
          Upload imagery, run semantic segmentation, compare outputs, and inspect overlays on an
          interactive map.
        </p>
        <span className="api-tag">Backend: {API_BASE_URL}</span>
      </header>

      <section className="workspace-grid">
        <UploadDropzone
          file={file}
          previewUrl={previewUrl}
          onFileSelected={onFileSelected}
          disabled={isLoading}
        />

        <section className="panel controls-panel">
          <div className="panel-head">
            <h2>Inference Controls</h2>
            <p>Adjust speed, quality, and map display behavior.</p>
          </div>

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
              onChange={(event) => setTileSize(Number(event.target.value) || 256)}
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

          <button onClick={runPrediction} className="primary-btn" disabled={isLoading}>
            {isLoading ? 'Running segmentation...' : 'Run Segmentation'}
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
          originalSrc={result?.images_base64?.original || previewUrl}
          predictionSrc={result?.images_base64?.prediction}
        />

        <section className="panel controls-panel map-controls-panel">
          <div className="panel-head">
            <h2>Map Controls</h2>
            <p>Control layer visibility and export your results.</p>
          </div>

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

      <MapOverlayView
        originalSrc={mapOriginalSrc}
        predictionSrc={mapPredictionSrc}
        overlayOpacity={mapOpacity}
        activeLayer={activeLayer}
        geoBounds={result?.geo?.bounds_epsg4326}
        shape={result?.shape}
      />
    </main>
  )
}

export default DashboardPage
