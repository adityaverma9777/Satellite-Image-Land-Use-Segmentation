import { useRef, useState } from 'react'

function UploadDropzone({ file, previewUrl, onFileSelected, disabled }) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)

    const dropped = event.dataTransfer.files?.[0]
    if (dropped) {
      onFileSelected(dropped)
    }
  }

  const handleInputChange = (event) => {
    const selected = event.target.files?.[0]
    if (selected) {
      onFileSelected(selected)
    }
  }

  return (
    <section className="panel upload-panel">
      <div className="panel-head">
        <h2>Upload Satellite Image</h2>
        <p>Drop GeoTIFF/JPG/PNG to run land-use segmentation.</p>
      </div>

      <div
        className={`dropzone ${isDragging ? 'is-dragging' : ''} ${disabled ? 'is-disabled' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          if (!disabled) {
            setIsDragging(true)
          }
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(event) => {
          if ((event.key === 'Enter' || event.key === ' ') && !disabled) {
            inputRef.current?.click()
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*,.tif,.tiff,.geotiff"
          className="visually-hidden"
          onChange={handleInputChange}
          disabled={disabled}
        />
        <p className="dropzone-title">Drag image here</p>
        <p className="dropzone-subtitle">or click to browse local files</p>
      </div>

      <div className="upload-meta">
        <strong>{file ? file.name : 'No file selected'}</strong>
      </div>

      {previewUrl ? (
        <div className="preview-frame">
          <img src={previewUrl} alt="Uploaded satellite preview" />
        </div>
      ) : (
        <div className="preview-empty">Image preview will appear here.</div>
      )}
    </section>
  )
}

export default UploadDropzone
