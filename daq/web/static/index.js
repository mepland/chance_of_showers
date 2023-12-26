/* global Plotly, io */

// get div by ID
const datetimeBoxDiv = document.getElementById('datetimeBox')
const pressureValueBoxDiv = document.getElementById('pressureValueBox')
const hadFlowBoxDiv = document.getElementById('hadFlowBox')

const pressureValueNormalizedGaugeDiv = document.getElementById('pressureValueNormalizedGauge')

const meanDiv = document.getElementById('meanDiv')
const liveDiv = document.getElementById('liveDiv')

// colors
const c0 = '#012169'
const c1 = '#993399'
const cGrey = '#7f7f7f'
const cYellow = '#FFD960'
const cOrange = '#E89923'

// markers
const msFlow0 = 'bowtie'
const msFlow1 = 'bowtie-open'
const mcFlow0 = c0
const mcFlow1 = c1
const markerSizeLarge = 12
const markerSizeSmall = 6

// flow null traces for legend entries
const legendEntryTraceFlow0 = {
  x: [null],
  y: [null],
  name: 'No Flow',
  type: 'scatter',
  mode: 'markers',
  marker: {
    size: markerSizeLarge,
    line: {
      width: 1.5,
      color: mcFlow0
    },
    symbol: msFlow0,
    color: mcFlow0
  }
}
const legendEntryTraceFlow1 = {
  x: [null],
  y: [null],
  name: 'Had Flow',
  type: 'scatter',
  mode: 'markers',
  marker: {
    size: markerSizeLarge,
    line: {
      width: 1.5,
      color: mcFlow1
    },
    symbol: msFlow1,
    color: mcFlow1
  }
}

// config for all graphs
const graphConfig = {
  displayModeBar: false,
  responsive: true
}

// boxes
function updateBoxes (tStr, iPolling, pressureValue, hadFlow) {
  datetimeBoxDiv.innerHTML = tStr.substr(0, tStr.length - 3) + ' i=' + iPolling.toString().padStart(2, '0')
  pressureValueBoxDiv.innerHTML = pressureValue
  hadFlowBoxDiv.innerHTML = hadFlow
}

// gauge
const gaugeLayout = {
  title: {
    text: 'Live Pressure',
    align: 'center'
  },
  xaxis: { visible: false },
  yaxis: { visible: false },
  shapes: [{
    type: 'line',
    // xref="x", x0=x1=100 doesn't work with gauge, use xref="paper" and adjust by eye
    xref: 'paper',
    x0: 0.475,
    x1: 0.475,
    yref: 'paper',
    y0: 0.1,
    y1: 0.9,
    line: {
      color: cGrey,
      width: 2,
      dash: 'dash'
    }
  }],
  // let width be set automatically
  height: 118,
  autosize: true,
  font: {
    color: cGrey,
    size: 12
  },
  margin: {
    t: 30,
    b: 15,
    l: 0,
    r: 0,
    p: 0
  }
}

const pressureValueNormalizedGaugeData = [
  {
    type: 'indicator',
    mode: 'gauge+number',
    domain: {
      x: [0.1, 0.9],
      y: [0.1, 0.9]
    },
    value: 0,
    number: {
      suffix: '%',
      valueformat: '.2f'
    },
    gauge: {
      shape: 'bullet',
      axis: {
        range: [0, 160],
        dtick: 40,
        tickwidth: 1,
        tickcolor: cGrey
      },
      borderwidth: 1,
      bordercolor: cGrey,
      bar: {
        color: cGrey,
        thickness: 0.4
      },
      steps: [
        { range: [0, 40], color: cOrange },
        { range: [40, 60], color: cYellow }
      ]
      /*
      // easy to set in axis coordinates, but dash doesn't work
      // use shapes on page instead
      threshold: {
        line: {
          color: "red",
          dash: "dash",
        },
        thickness: 1,
        value: 100,
      },
      */
    }
  }
]

Plotly.newPlot(pressureValueNormalizedGaugeDiv, pressureValueNormalizedGaugeData, gaugeLayout, graphConfig)

function updateGauge (pressureValueNormalized) {
  const pressureValueNormalizedUpdate = {
    value: pressureValueNormalized
  }

  Plotly.update(pressureValueNormalizedGaugeDiv, pressureValueNormalizedUpdate)
}

// traces
const meanLayout = {
  xaxis: {
    title: '1 Min Samples',
    zeroline: false
  },
  yaxis: {
    title: 'Mean Pressure %',
    zeroline: false,
    hoverformat: '.2f'

  },
  colorway: [c0],
  showlegend: true,
  legend: {
    orientation: 'h',
    xanchor: 'center',
    yanchor: 'bottom',
    x: 0.5,
    y: 1.0
  },
  shapes: [{
    type: 'line',
    xref: 'paper',
    x0: 0,
    x1: 1,
    yref: 'y',
    y0: 100,
    y1: 100,
    line: {
      color: cGrey,
      width: 2,
      dash: 'dash'
    }
  }],
  autosize: true,
  margin: {
    t: 0,
    b: 70,
    l: 70,
    r: 0,
    p: 0
  },
  font: {
    color: cGrey,
    size: 14
  }
}

const meanTrace = {
  x: [],
  y: [],
  customdata: [],
  type: 'scatter',
  mode: 'lines+markers',
  marker: {
    size: markerSizeLarge,
    line: {
      width: 1.5
    },
    symbol: []
  },
  showlegend: false,
  hovertemplate:
    '1 Min Sample: %{x:%Y-%m-%d %H:%M:%S}<br>' +
    'Mean Pressure: %{y:.2f}%<br>' +
    'Had Flow: %{customdata:d}' +
    '<extra></extra>'
}

const liveLayout = JSON.parse(JSON.stringify(meanLayout))
const liveTrace = JSON.parse(JSON.stringify(meanTrace))

liveLayout.xaxis.title = 'Polling Samples'
liveLayout.xaxis.range = [-2.5, 62.5]
liveLayout.yaxis.title = 'Pressure %'
liveLayout.showlegend = false
liveTrace.mode = 'markers'
liveTrace.marker.size = markerSizeSmall
liveTrace.hovertemplate =
    'i: %{x:d}<br>' +
    'Pressure: %{y:.2f}%<br>' +
    'Flow: %{customdata:d}' +
    '<extra></extra>'

Plotly.newPlot(
  meanDiv,
  [meanTrace, legendEntryTraceFlow0, legendEntryTraceFlow1],
  meanLayout,
  graphConfig
)

Plotly.newPlot(
  liveDiv,
  [liveTrace, legendEntryTraceFlow0, legendEntryTraceFlow1],
  liveLayout,
  graphConfig
)

const liveXArray = []
const liveYArray = []
const liveZArray = []
const liveMSArray = []
const liveMCArray = []

function updateChart (div, xArray, yArray, zArray, msArray, mcArray, maxGraphPoints, markerSize = markerSizeLarge, xValue = null, yValue = null, zValue = null) {
  if (maxGraphPoints > 0) {
    if (maxGraphPoints <= xArray.length) {
      xArray.shift()
      yArray.shift()
      zArray.shift()
      msArray.shift()
      mcArray.shift()
    }
    xArray.push(xValue)
    yArray.push(yValue)
    zArray.push(zValue)

    if (zValue !== 1) {
      msArray.push(msFlow0)
      mcArray.push(mcFlow0)
    } else {
      msArray.push(msFlow1)
      mcArray.push(mcFlow1)
    }
  }

  const dataUpdate = {
    x: [xArray],
    y: [yArray],
    customdata: [zArray],
    marker: {
      size: markerSize,
      line: {
        width: 1.5,
        color: mcArray
      },
      symbol: msArray,
      color: mcArray
    }
  }

  Plotly.update(div, dataUpdate, {}, [0])
}

// update everything each msg
function updateAll (data) {
  const tStr = data.tLocalStr
  const iPolling = parseInt(data.iPolling)
  const pressureValue = parseInt(data.pressureValue)
  const pressureValueNormalized = (100 * parseFloat(data.pressureValueNormalized)).toFixed(2)
  const hadFlow = parseInt(data.hadFlow)

  updateBoxes(tStr, iPolling, pressureValue, hadFlow)

  updateGauge(pressureValueNormalized)

  let updatedMean = false
  for (const prop in data) {
    if (prop === 'tLocalStrNLast') {
      updatedMean = true
      break
    }
  }

  if (updatedMean) {
    const tStrNLast = data.tLocalStrNLast
    const meanPressureValueNormalizedNLast = data.meanPressureValueNormalizedNLast
    const pastHadFlowNLast = data.pastHadFlowNLast
    const msNLast = []
    const mcNLast = []

    for (let i = 0; i < meanPressureValueNormalizedNLast.length; i++) {
      meanPressureValueNormalizedNLast[i] = (100 * parseFloat(meanPressureValueNormalizedNLast[i])).toFixed(2)
      pastHadFlowNLast[i] = parseInt(pastHadFlowNLast[i])

      if (pastHadFlowNLast[i] !== 1) {
        msNLast[i] = msFlow0
        mcNLast[i] = mcFlow0
      } else {
        msNLast[i] = msFlow1
        mcNLast[i] = mcFlow1
      }
    }

    updateChart(
      meanDiv,
      tStrNLast,
      meanPressureValueNormalizedNLast,
      pastHadFlowNLast,
      msNLast,
      mcNLast,
      -1
    )
  }

  updateChart(
    liveDiv,
    liveXArray,
    liveYArray,
    liveZArray,
    liveMSArray,
    liveMCArray,
    60,
    markerSizeSmall,
    iPolling,
    pressureValueNormalized,
    hadFlow
  )
}

// SocketIO Code
const socket = io.connect()

// receive data from server
socket.on('emitData', function (msg) {
  const data = JSON.parse(msg)
  // console.log(msg);
  updateAll(data)
})
