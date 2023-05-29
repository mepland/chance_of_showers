// get div by ID
var datetime_box_div = document.getElementById("datetime_box");
var pressure_value_box_div = document.getElementById("pressure_value_box");
var had_flow_box_div = document.getElementById("had_flow_box");

var pressure_value_normalized_gauge_div = document.getElementById("pressure_value_normalized_gauge");

var mean_pressure_value_normalized_trace_div = document.getElementById("mean_pressure_value_normalized_trace");
var past_had_flow_trace_div = document.getElementById("past_had_flow_trace");

var pressure_value_normalized_trace_div = document.getElementById("pressure_value_normalized_trace");
var had_flow_trace_div = document.getElementById("had_flow_trace");

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
    // domain: { x: [0, 1], y: [0, 1] },
    domain: {x: [0.1, 0.9], y: [0.1, 1]},
    value: 0,
    title: { text: "%" },
    type: "indicator",
    mode: "gauge+number",
    delta: { reference: 100 },
    gauge: {
      shape: "bullet",
      axis: { range: [0, 150] },
      steps: [
        { range: [0, 50], color: "lightsalmon" },
      ],
    },
  },
];

var gauge_layout = { width: 350, height: 250, margin: { t: 0, b: 0, l: 0, r: 0 } };

Plotly.newPlot(pressure_value_normalized_gauge_div, pressure_value_normalized_gauge_data, gauge_layout, graph_config);

function updateGauge(pressure_value_normalized) {
  var pressure_value_normalized_update = {
    value: pressure_value_normalized,
  };

  Plotly.update(pressure_value_normalized_gauge_div, pressure_value_normalized_update);
}

// traces
var mean_pressure_value_normalized_trace = {
  x: [],
  y: [],
  name: "Mean Pressure %",
  // mode: "lines+markers",
  mode: "markers",
  type: "line",
};
var past_had_flow_trace = {
  x: [],
  y: [],
  name: "Past Had Flow",
  // mode: "lines+markers",
  mode: "markers",
  type: "line",
};
var pressure_value_normalized_trace = {
  x: [],
  y: [],
  name: "Pressure %",
  // mode: "lines+markers",
  mode: "markers",
  type: "line",
};
var had_flow_trace = {
  x: [],
  y: [],
  name: "Had Flow",
  // mode: "lines+markers",
  mode: "markers",
  type: "line",
};

var pressure_value_mean_trace_layout = {
  autosize: true,
  title: {
    text: "Mean Pressure %",
  },
  font: {
    size: 14,
    color: "#7f7f7f",
  },
  colorway: ["#B22222"],
  margin: { t: 30, b: 20, l: 30, r: 20, pad: 0 },
};
var past_had_flow_trace_layout = {
  autosize: true,
  title: {
    text: "Past Had Flow",
  },
  font: {
    size: 14,
    color: "#7f7f7f",
  },
  colorway: ["#00008B"],
  margin: { t: 30, b: 20, l: 30, r: 20, pad: 0 },
};
var pressure_value_polling_trace_layout = {
  autosize: true,
  title: {
    text: "Pressure %",
  },
  font: {
    size: 14,
    color: "#7f7f7f",
  },
  colorway: ["#B22222"],
  margin: { t: 30, b: 20, l: 30, r: 20, pad: 0 },
};
var had_flow_trace_layout = {
  autosize: true,
  title: {
    text: "Had Flow",
  },
  font: {
    size: 14,
    color: "#7f7f7f",
  },
  colorway: ["#00008B"],
  margin: { t: 30, b: 20, l: 30, r: 20, pad: 0 },
};

Plotly.newPlot(
  mean_pressure_value_normalized_trace_div,
  [mean_pressure_value_normalized_trace],
  pressure_value_mean_trace_layout,
  graph_config
);
Plotly.newPlot(
  past_had_flow_trace_div,
  [past_had_flow_trace],
  past_had_flow_trace_layout,
  graph_config
);
Plotly.newPlot(
  pressure_value_normalized_trace_div,
  [pressure_value_normalized_trace],
  pressure_value_polling_trace_layout,
  graph_config
);
Plotly.newPlot(
  had_flow_trace_div,
  [had_flow_trace],
  had_flow_trace_layout,
  graph_config
);

let mean_pressure_value_normalized_x_array = [];
let mean_pressure_value_normalized_y_array = [];
let past_had_flow_x_array = [];
let past_had_flow_y_array = [];
let pressure_value_normalized_x_array = [];
let pressure_value_normalized_y_array = [];
let had_flow_x_array = [];
let had_flow_y_array = [];

function updateChart(lineChartDiv, xArray, yArray, xValue, yValue, max_graph_points) {
  if (xArray.length >= max_graph_points) {
    xArray.shift();
  }
  if (yArray.length >= max_graph_points) {
    yArray.shift();
  }
  xArray.push(xValue)
  yArray.push(yValue);

  var data_update = {
    x: [xArray],
    y: [yArray],
  };

  Plotly.update(lineChartDiv, data_update);
}

// update everything each msg
function updateAll(jsonResponse) {
  let t_str = jsonResponse.t_est_str;
  let i_polling = parseInt(jsonResponse.i_polling);
  let pressure_value = parseInt(jsonResponse.pressure_value);
  let pressure_value_normalized = (100*parseFloat(jsonResponse.pressure_value_normalized)).toFixed(2);
  let had_flow = parseInt(jsonResponse.had_flow);

  let mean_pressure_value_normalized = (100*parseFloat(jsonResponse.mean_pressure_value_normalized)).toFixed(2);
  let past_had_flow = parseInt(jsonResponse.past_had_flow);

  updateBoxes(t_str, i_polling, pressure_value, had_flow);

  updateGauge(pressure_value_normalized);

  if (i_polling = 0) {
    updateChart(
      mean_pressure_value_normalized_trace_div,
      mean_pressure_value_normalized_x_array,
      mean_pressure_value_normalized_y_array,
      t_str,
      mean_pressure_value_normalized,
      15 
    );
    updateChart(
      past_had_flow_trace_div,
      past_had_flow_x_array,
      past_had_flow_y_array,
      t_str,
      past_had_flow,
      15 
    );
  }

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
}

// SocketIO Code
// var socket = io.connect("http://" + document.domain + ":" + location.port);

var socket = io.connect();

// receive data from server
socket.on("updateData", function (msg) {
  var data = JSON.parse(msg);
  // write data to console for debugging
  // console.log(msg);
  updateAll(data);
});
