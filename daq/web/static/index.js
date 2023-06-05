// get div by ID
var datetime_box_div = document.getElementById("datetime_box");
var pressure_value_box_div = document.getElementById("pressure_value_box");
var had_flow_box_div = document.getElementById("had_flow_box");

var pressure_value_normalized_gauge_div = document.getElementById("pressure_value_normalized_gauge");

var mean_div = document.getElementById("mean_div");
var live_div = document.getElementById("live_div");

// colors
var c0 = "#012169";
// var c1 = "#993399";
var c_grey = "#7f7f7f";

// markers
var marker_symbol_flow_0 = "bowtie";
var marker_symbol_flow_1 = "bowtie-open";
var marker_size_large = 12;
var marker_size_small = 8;

// flow null traces
var flow_0 = {
  x: [null],
  y: [null],
  name: "No Flow",
  type: "scatter",
  mode: "markers",
  marker: {
    size: marker_size_large,
    line: {width: 1.5},
    symbol: marker_symbol_flow_0,
  },
};
var flow_1 = {
  x: [null],
  y: [null],
  name: "Had Flow",
  type: "scatter",
  mode: "markers",
  marker: {
    size: marker_size_large,
    line: {width: 1.5},
    symbol: marker_symbol_flow_1,
  },
};

// config for all graphs
var graph_config = {
  displayModeBar: false,
  responsive: true,
};

// boxes
function updateBoxes(t_str, i_polling, pressure_value, had_flow) {
  datetime_box_div.innerHTML = t_str.substr(0,t_str.length-3) + ", i = " + i_polling;
  pressure_value_box_div.innerHTML = pressure_value;
  had_flow_box_div.innerHTML = had_flow;
}

// gauge
var pressure_value_normalized_gauge_data = [
  {
    domain: {x: [0.1, 0.9], y: [0.2, 1]},
    value: 0,
    title: { text: "%" },
    type: "indicator",
    mode: "gauge+number",
    delta: { reference: 100 },
    gauge: {
      shape: "bullet",
      axis: { range: [0, 150] },
      bar: {color: "#262626"},
      steps: [
        { range: [0, 50], color: "#C84E00" },
      ],
    },
  },
];

var gauge_layout = { width: 400, height: 100, margin: { t: 0, b: 0, l: 0, r: 0 } };

Plotly.newPlot(pressure_value_normalized_gauge_div, pressure_value_normalized_gauge_data, gauge_layout, graph_config);

function updateGauge(pressure_value_normalized) {
  var pressure_value_normalized_update = {
    value: pressure_value_normalized,
  };

  Plotly.update(pressure_value_normalized_gauge_div, pressure_value_normalized_update);
}

// trace layout
var mean_layout = {
  xaxis: {title: "1 Min Samples"},
  yaxis: {title: "Mean Pressure %"},
  colorway: [c0],
  showlegend: true,
  legend: {
    orientation: "h",
    xanchor: "center",
    yanchor: "bottom",
    x: 0.5,
    y: 1.0,
  },
  autosize: true,
  margin: {t: 0, b: 70, l: 55, r: 0},
  font: {color: c_grey, size: 14},
};

/*
var pressure_value_normalized_trace = {
  x: [],
  y: [],
  name: "Pressure %",
  mode: "markers",
  type: "line",
};
var had_flow_trace = {
  x: [],
  y: [],
  name: "Had Flow",
  mode: "markers",
  type: "line",
};

var pressure_value_trace_layout = {
  autosize: true,
  title: {
    text: "Pressure %",
  },
  font: {
    size: 14,
    color: c_grey,
  },
  colorway: [c0],
  margin: { t: 30, b: 20, l: 40, r: 20, pad: 0 },
};
var had_flow_trace_layout = {
  autosize: true,
  title: {
    text: "Had Flow",
  },
  font: {
    size: 14,
    color: c_grey,
  },
  colorway: [c1],
  margin: { t: 30, b: 20, l: 30, r: 20, pad: 0 },
};

Plotly.newPlot(
  pressure_value_normalized_trace_div,
  [pressure_value_normalized_trace],
  pressure_value_trace_layout,
  graph_config
);
Plotly.newPlot(
  had_flow_trace_div,
  [had_flow_trace],
  had_flow_trace_layout,
  graph_config
);

let pressure_value_normalized_x_array = [];
let pressure_value_normalized_y_array = [];
let had_flow_x_array = [];
let had_flow_y_array = [];

function updateChart(lineChartDiv, xArray, yArray, xValue, yValue, max_graph_points) {
  if ( 0 < max_graph_points) {
    if (xArray.length >= max_graph_points) {
      xArray.shift();
    }
    if (yArray.length >= max_graph_points) {
      yArray.shift();
    }
    xArray.push(xValue);
    yArray.push(yValue);
  }

  var data_update = {
    x: [xArray],
    y: [yArray],
  };

  Plotly.update(lineChartDiv, data_update);
}
*/

// update everything each msg
function updateAll(data) {
  let t_str = data.t_est_str;
  let i_polling = parseInt(data.i_polling);
  let pressure_value = parseInt(data.pressure_value);
  let pressure_value_normalized = (100*parseFloat(data.pressure_value_normalized)).toFixed(2);
  let had_flow = parseInt(data.had_flow);

  updateBoxes(t_str, i_polling, pressure_value, had_flow);

  updateGauge(pressure_value_normalized);

  var updated_mean = false;
  for (const prop in data) {
    if (prop == "t_est_str_n_last") {
       updated_mean = true;
       break;
    }
  }

  if (updated_mean) {
    let t_str_n_last = data.t_est_str_n_last
    let mean_pressure_value_normalized_n_last = data.mean_pressure_value_normalized_n_last
    let past_had_flow_n_last = data.past_had_flow_n_last
    let past_had_flow_n_last_symbol = []

    for (var i = 0; i < mean_pressure_value_normalized_n_last.length; i++) {
      mean_pressure_value_normalized_n_last[i] = (100*parseFloat(mean_pressure_value_normalized_n_last[i])).toFixed(2);
      past_had_flow_n_last[i] = parseInt(past_had_flow_n_last[i]);

      // TODo testing
      if (i < 5) {
        past_had_flow_n_last[i] = 1;
      } // TODo end testing

      if (past_had_flow_n_last[i] == 0) {
        past_had_flow_n_last_symbol[i] = marker_symbol_flow_0;
      }
      else {
        past_had_flow_n_last_symbol[i] = marker_symbol_flow_1;
      }
    }

    var mean_pressure_value_normalized_trace = {
      x: t_str_n_last,
      y: mean_pressure_value_normalized_n_last,
      type: "scatter",
      mode: "lines+markers",
      marker: {
        size: marker_size_large,
		line: {width: 1.5},
        symbol: past_had_flow_n_last_symbol,
      },
      showlegend: false, 
    };

    Plotly.react(mean_div, [mean_pressure_value_normalized_trace, flow_0, flow_1], mean_layout, graph_config);
  }
/*
  updateChart(
    pressure_value_normalized_trace_div,
    pressure_value_normalized_x_array,
    pressure_value_normalized_y_array,
    i_polling,
    pressure_value_normalized,
    60
  );

  updateChart(
    had_flow_trace_div,
    had_flow_x_array,
    had_flow_y_array,
    i_polling,
    had_flow,
    60
  );
*/
}

// SocketIO Code
// var socket = io.connect("http://" + document.domain + ":" + location.port);

var socket = io.connect();

// receive data from server
socket.on("emit_data", function (msg) {
  var data = JSON.parse(msg);
  // write data to console for debugging
  // console.log(msg);
  updateAll(data);
});
