import React from 'react'
import PlotlyComponent from 'react-plotly.js'

// Vite sometimes imports CommonJS default exports as an object { default: Component }
const Plot = PlotlyComponent && PlotlyComponent.default ? PlotlyComponent.default : PlotlyComponent

export default function PlotlyChart(props) {
  if (!Plot || typeof Plot !== 'function' && typeof Plot !== 'object') {
    return <div style={{height: props.layout?.height || 260}}>Loading Chart...</div>
  }
  return <Plot {...props} />
}
