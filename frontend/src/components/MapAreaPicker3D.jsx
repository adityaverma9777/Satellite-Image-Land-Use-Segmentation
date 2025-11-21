import { useEffect, useMemo, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { makeApiUrl, makeTemplateApiUrl } from '../config'

const EMPTY_COLLECTION = {
  type: 'FeatureCollection',
  features: [],
}

const EMPTY_POINT_COLLECTION = {
  type: 'FeatureCollection',
  features: [],
}

const MIN_BBOX_SPAN = 0.0001

function isValidBounds(bounds) {
  return (
    Array.isArray(bounds) &&
    bounds.length === 2 &&
    Array.isArray(bounds[0]) &&
    Array.isArray(bounds[1]) &&
    bounds[0].length === 2 &&
    bounds[1].length === 2
  )
}

function normalizeBounds(a, b) {
  return {
    south: Math.min(a.lat, b.lat),
    west: Math.min(a.lng, b.lng),
    north: Math.max(a.lat, b.lat),
    east: Math.max(a.lng, b.lng),
  }
}

function boundsToGeoJson(bounds) {
  const { south, west, north, east } = bounds
  return {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [west, north],
              [east, north],
              [east, south],
              [west, south],
              [west, north],
            ],
          ],
        },
      },
    ],
  }
}

function overlayCoordinatesFromBounds(bounds) {
  const south = bounds[0][0]
  const west = bounds[0][1]
  const north = bounds[1][0]
  const east = bounds[1][1]

  return [
    [west, north],
    [east, north],
    [east, south],
    [west, south],
  ]
}

function buildSatelliteStyle(attributionText) {
  return {
    version: 8,
    sources: {
      satellite: {
        type: 'raster',
        tiles: [makeTemplateApiUrl('/map/tile/{z}/{x}/{y}?layer=satellite')],
        tileSize: 256,
        attribution: attributionText,
      },
    },
    layers: [
      {
        id: 'satellite-layer',
        type: 'raster',
        source: 'satellite',
      },
    ],
  }
}

function pointToGeoJson(lngLat) {
  if (!lngLat) {
    return EMPTY_POINT_COLLECTION
  }

  return {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [lngLat.lng, lngLat.lat],
        },
      },
    ],
  }
}

function MapAreaPicker3D({
  onSelectionChange,
  overlayImageUrl,
  overlayBounds,
  overlayOpacity,
}) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const firstCornerRef = useRef(null)
  const isSelectingRef = useRef(false)
  const draftBoundsRef = useRef(null)

  const [mapStatus, setMapStatus] = useState(null)
  const [mapError, setMapError] = useState('')
  const [isMapReady, setIsMapReady] = useState(false)
  const [isSelecting, setIsSelecting] = useState(false)
  const [draftBounds, setDraftBounds] = useState(null)
  const [selectionMessage, setSelectionMessage] = useState(
    'Click Start Area Selection, then click two map points.',
  )
  const [selectionError, setSelectionError] = useState('')
  const [cameraZoom, setCameraZoom] = useState(2.2)

  useEffect(() => {
    let isActive = true

    const loadStatus = async () => {
      try {
        const response = await fetch(makeApiUrl('/map/status'))
        const data = await response.json()
        if (!response.ok) {
          throw new Error(data.detail || 'Failed to load map status.')
        }
        if (isActive) {
          setMapStatus(data)
          setMapError('')
        }
      } catch (error) {
        if (isActive) {
          setMapError(error.message || 'Could not connect to offline map API.')
        }
      }
    }

    loadStatus()
    return () => {
      isActive = false
    }
  }, [])

  const statusText = useMemo(() => {
    if (mapError) {
      return mapError
    }
    if (!mapStatus) {
      return 'Loading map status...'
    }
    if (!mapStatus.satellite_ready) {
      return 'No satellite source is available. Add offline MBTiles or enable online fallback.'
    }

    if (mapStatus.active_source === 'online') {
      if (mapStatus.terrain_ready) {
        return 'Using free online satellite and terrain services.'
      }
      return 'Using free online satellite service. Terrain is unavailable.'
    }

    if (!mapStatus.terrain_ready) {
      return 'Using offline satellite MBTiles. Add terrain-rgb.mbtiles to enable 3D terrain.'
    }
    return 'Using offline satellite and terrain MBTiles.'
  }, [mapError, mapStatus])

  useEffect(() => {
    if (!containerRef.current || mapRef.current || !mapStatus) {
      return
    }

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: buildSatelliteStyle(mapStatus.satellite_attribution || 'Satellite basemap'),
      center: [78.9629, 20.5937],
      zoom: 2.2,
      pitch: 0,
      bearing: 0,
      maxPitch: 0,
      antialias: true,
      attributionControl: true,
    })

    map.addControl(new maplibregl.NavigationControl(), 'top-right')

    map.dragRotate.disable()
    map.touchZoomRotate.disableRotation()

    const seenErrors = new Set()
    map.on('error', (event) => {
      const message = event?.error?.message || event?.error?.toString?.() || 'Map tile/source error.'
      if (seenErrors.has(message)) {
        return
      }
      seenErrors.add(message)
      console.warn('[map]', message)
      setMapError((current) => current || 'Some map tiles failed to load. You can still select an area.')
    })

    const updateCameraZoomState = () => {
      setCameraZoom(Number(map.getZoom().toFixed(2)))
    }
    map.on('zoomend', updateCameraZoomState)
    map.on('moveend', updateCameraZoomState)

    map.on('load', () => {
      setIsMapReady(true)

      map.addSource('selection-source', {
        type: 'geojson',
        data: EMPTY_COLLECTION,
      })

      map.addLayer({
        id: 'selection-fill',
        type: 'fill',
        source: 'selection-source',
        paint: {
          'fill-color': '#12c2e9',
          'fill-opacity': 0.22,
        },
      })

      map.addLayer({
        id: 'selection-outline',
        type: 'line',
        source: 'selection-source',
        paint: {
          'line-color': '#12c2e9',
          'line-width': 2,
        },
      })

      map.addSource('first-corner-source', {
        type: 'geojson',
        data: EMPTY_POINT_COLLECTION,
      })

      map.addLayer({
        id: 'first-corner-dot',
        type: 'circle',
        source: 'first-corner-source',
        paint: {
          'circle-radius': 6,
          'circle-color': '#12c2e9',
          'circle-stroke-width': 2,
          'circle-stroke-color': '#ffffff',
        },
      })

      if (mapStatus.terrain_ready) {
        map.addSource('terrain-dem', {
          type: 'raster-dem',
          tiles: [makeTemplateApiUrl('/map/tile/{z}/{x}/{y}?layer=terrain')],
          tileSize: 256,
          maxzoom: 14,
          encoding: mapStatus.terrain_encoding || 'terrarium',
        })
        map.setTerrain({ source: 'terrain-dem', exaggeration: 1.25 })
      }

      updateCameraZoomState()
    })

    const updateSelectionPolygon = (bounds) => {
      const source = map.getSource('selection-source')
      if (!source) {
        return
      }

      source.setData(bounds ? boundsToGeoJson(bounds) : EMPTY_COLLECTION)
    }

    const updateFirstCornerMarker = (lngLat) => {
      const source = map.getSource('first-corner-source')
      if (!source) {
        return
      }

      source.setData(pointToGeoJson(lngLat))
    }

    map.on('click', (event) => {
      if (!isSelectingRef.current) {
        return
      }

      if (!firstCornerRef.current) {
        firstCornerRef.current = event.lngLat
        updateFirstCornerMarker(event.lngLat)
        setSelectionError('')
        setSelectionMessage('First corner set. Move mouse and click second corner.')
        return
      }

      const bounds = normalizeBounds(firstCornerRef.current, event.lngLat)

      if (bounds.north - bounds.south < MIN_BBOX_SPAN || bounds.east - bounds.west < MIN_BBOX_SPAN) {
        setSelectionError('Selected area is too small. Click the second corner farther away.')
        return
      }

      firstCornerRef.current = null
      updateFirstCornerMarker(null)
      updateSelectionPolygon(bounds)
      const selection = {
        ...bounds,
        map_zoom: Number(map.getZoom().toFixed(2)),
      }
      draftBoundsRef.current = selection
      setDraftBounds(selection)
      setIsSelecting(false)
      isSelectingRef.current = false
      setSelectionError('')
      setSelectionMessage('Area drafted. Click Confirm Selected Area.')
    })

    map.on('mousemove', (event) => {
      if (!isSelectingRef.current || !firstCornerRef.current) {
        return
      }

      const previewBounds = normalizeBounds(firstCornerRef.current, event.lngLat)
      updateSelectionPolygon(previewBounds)
    })

    mapRef.current = map

    return () => {
      setIsMapReady(false)
      map.remove()
      mapRef.current = null
    }
  }, [mapStatus, onSelectionChange])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !isMapReady || !map.isStyleLoaded()) {
      return
    }

    const hasLayer = Boolean(map.getLayer('analysis-overlay'))

    if (!overlayImageUrl || !isValidBounds(overlayBounds)) {
      if (hasLayer) {
        map.removeLayer('analysis-overlay')
      }
      if (map.getSource('analysis-overlay-source')) {
        map.removeSource('analysis-overlay-source')
      }
      return
    }

    const coordinates = overlayCoordinatesFromBounds(overlayBounds)

    if (hasLayer) {
      map.removeLayer('analysis-overlay')
    }
    if (map.getSource('analysis-overlay-source')) {
      map.removeSource('analysis-overlay-source')
    }

    map.addSource('analysis-overlay-source', {
      type: 'image',
      url: overlayImageUrl,
      coordinates,
    })

    map.addLayer({
      id: 'analysis-overlay',
      type: 'raster',
      source: 'analysis-overlay-source',
      paint: {
        'raster-opacity': overlayOpacity,
      },
    })
  }, [overlayImageUrl, overlayBounds, overlayOpacity, isMapReady])

  const startSelection = () => {
    const map = mapRef.current
    firstCornerRef.current = null
    draftBoundsRef.current = null
    setDraftBounds(null)
    setIsSelecting(true)
    isSelectingRef.current = true
    setSelectionError('')
    setSelectionMessage('Selection started. Click first corner on the map.')
    onSelectionChange(null)

    if (!map) {
      return
    }

    const areaSource = map.getSource('selection-source')
    if (areaSource) {
      areaSource.setData(EMPTY_COLLECTION)
    }

    const cornerSource = map.getSource('first-corner-source')
    if (cornerSource) {
      cornerSource.setData(EMPTY_POINT_COLLECTION)
    }
  }

  const confirmSelection = () => {
    if (!draftBoundsRef.current) {
      setSelectionError('Draw an area first, then confirm it.')
      return
    }

    onSelectionChange(draftBoundsRef.current)
    setSelectionError('')
    setSelectionMessage('Area confirmed. You can now run segmentation.')
  }

  const clearSelection = () => {
    const map = mapRef.current
    firstCornerRef.current = null
    draftBoundsRef.current = null
    setDraftBounds(null)
    setIsSelecting(false)
    isSelectingRef.current = false
    setSelectionError('')
    setSelectionMessage('Selection cleared. Click Start Area Selection to begin again.')
    onSelectionChange(null)

    if (!map) {
      return
    }

    const areaSource = map.getSource('selection-source')
    if (areaSource) {
      areaSource.setData(EMPTY_COLLECTION)
    }

    const cornerSource = map.getSource('first-corner-source')
    if (cornerSource) {
      cornerSource.setData(EMPTY_POINT_COLLECTION)
    }
  }

  return (
    <section className="panel world-map-panel">
      <div className="panel-head">
        <h2>World Map Explorer</h2>
        <p>
          Satellite map with terrain support. Start selection, click two corners, confirm area,
          then run segmentation.
        </p>
      </div>

      <p className="map-status-text">{statusText}</p>

      <p className="selection-help-text">{selectionMessage}</p>

      <p className="selection-coords-text">Camera Zoom: {cameraZoom.toFixed(2)} (top-down selection)</p>

      {draftBounds ? (
        <p className="selection-coords-text">
          Draft: {draftBounds.south.toFixed(4)}, {draftBounds.west.toFixed(4)} to{' '}
          {draftBounds.north.toFixed(4)}, {draftBounds.east.toFixed(4)}
        </p>
      ) : null}

      {selectionError ? <p className="error-text">{selectionError}</p> : null}

      <div className="map-actions-row">
        <button className={`toggle-btn ${isSelecting ? 'is-active' : ''}`} onClick={startSelection} type="button">
          {isSelecting ? 'Selecting...' : 'Start Area Selection'}
        </button>
        <button className="secondary-btn" onClick={confirmSelection} type="button">
          Confirm Selected Area
        </button>
        <button className="secondary-btn" onClick={clearSelection} type="button">
          Clear Selected Area
        </button>
      </div>

      <div ref={containerRef} className="map-canvas-3d" />
    </section>
  )
}

export default MapAreaPicker3D
