import { ImageOverlay, MapContainer, TileLayer } from 'react-leaflet'

function getFallbackBounds(shape) {
  const height = shape?.height || 1000
  const width = shape?.width || 1000
  const latSpan = Math.max(0.8, height / 700)
  const lngSpan = Math.max(0.8, width / 700)
  return [
    [18.0, -122.0],
    [18.0 + latSpan, -122.0 + lngSpan],
  ]
}

function centerFromBounds(bounds) {
  return [
    (bounds[0][0] + bounds[1][0]) / 2,
    (bounds[0][1] + bounds[1][1]) / 2,
  ]
}

function MapOverlayView({
  originalSrc,
  predictionSrc,
  overlayOpacity,
  activeLayer,
  geoBounds,
  shape,
}) {
  if (!originalSrc || !predictionSrc) {
    return (
      <section className="panel map-panel empty-state">
        <h2>Map Overlay</h2>
        <p>Prediction overlay will appear on an interactive map after inference.</p>
      </section>
    )
  }

  const hasGeoBounds =
    Array.isArray(geoBounds) &&
    geoBounds.length === 2 &&
    Array.isArray(geoBounds[0]) &&
    Array.isArray(geoBounds[1])

  const overlayBounds = hasGeoBounds ? geoBounds : getFallbackBounds(shape)
  const mapCenter = centerFromBounds(overlayBounds)
  const overlayImage = activeLayer === 'original' ? originalSrc : predictionSrc

  return (
    <section className="panel map-panel">
      <div className="panel-head">
        <h2>Interactive Map View</h2>
        <p>
          {hasGeoBounds
            ? 'Geo-referenced overlay from source metadata.'
            : 'No geospatial metadata detected: overlay shown on approximate map bounds.'}
        </p>
      </div>
      <div className="map-shell">
        <MapContainer center={mapCenter} zoom={11} scrollWheelZoom className="map-canvas">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <ImageOverlay bounds={overlayBounds} url={overlayImage} opacity={overlayOpacity} />
        </MapContainer>
      </div>
    </section>
  )
}

export default MapOverlayView
