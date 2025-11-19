import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider'

function BeforeAfterSlider({ originalSrc, predictionSrc }) {
  if (!originalSrc || !predictionSrc) {
    return (
      <section className="panel compare-panel empty-state">
        <h2>Before / After</h2>
        <p>Run prediction to compare original imagery and segmentation output.</p>
      </section>
    )
  }

  return (
    <section className="panel compare-panel">
      <div className="panel-head">
        <h2>Before / After Comparison</h2>
        <p>Slide horizontally to inspect model output quality.</p>
      </div>
      <div className="compare-shell">
        <ReactCompareSlider
          itemOne={<ReactCompareSliderImage src={originalSrc} alt="Original satellite image" />}
          itemTwo={<ReactCompareSliderImage src={predictionSrc} alt="Predicted segmentation mask" />}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </section>
  )
}

export default BeforeAfterSlider
