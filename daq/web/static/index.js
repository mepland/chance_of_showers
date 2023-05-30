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

var gauge_layout = { width: 400, height: 100, margin: { t: 0, b: 0, l: 0, r: 0 } };

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
  mode: "lines+markers",
  type: "line",
};
var past_had_flow_trace = {
  x: [],
  y: [],
  name: "Past Had Flow",
  mode: "lines+markers",
  type: "line",
};
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

var mean_pressure_value_trace_layout = {
  autosize: true,
  title: {
    text: "Mean Pressure %",
  },
  font: {
    size: 14,
    color: "#7f7f7f",
  },
  colorway: ["#B22222"],
  margin: { t: 30, b: 100, l: 40, r: 20, pad: 0 },
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
  margin: { t: 30, b: 100, l: 30, r: 20, pad: 0 },
};
var pressure_value_trace_layout = {
  autosize: true,
  title: {
    text: "Pressure %",
  },
  font: {
    size: 14,
    color: "#7f7f7f",
  },
  colorway: ["#B22222"],
  margin: { t: 30, b: 20, l: 40, r: 20, pad: 0 },
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
  mean_pressure_value_trace_layout,
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

// update everything each msg
function updateAll(jsonResponse) {
  let t_str = jsonResponse.t_est_str;
  let i_polling = parseInt(jsonResponse.i_polling);
  let pressure_value = parseInt(jsonResponse.pressure_value);
  let pressure_value_normalized = (100*parseFloat(jsonResponse.pressure_value_normalized)).toFixed(2);
  let had_flow = parseInt(jsonResponse.had_flow);


  updateBoxes(t_str, i_polling, pressure_value, had_flow);

  updateGauge(pressure_value_normalized);

  if (i_polling == 0) {
    let t_str_n_last = jsonResponse.t_est_str_n_last
    let mean_pressure_value_normalized_n_last = jsonResponse.mean_pressure_value_normalized_n_last
    let past_had_flow_n_last = jsonResponse.past_had_flow_n_last
    // use loops instead of map for compatibility
    for (var i = 0; i < mean_pressure_value_normalized_n_last.length; i++) {
      mean_pressure_value_normalized_n_last[i] = (100*parseFloat(mean_pressure_value_normalized_n_last[i])).toFixed(2);
    }
    for (var i = 0; i < past_had_flow_n_last.length; i++) {
      past_had_flow_n_last[i] = parseInt(past_had_flow_n_last[i]);
    }
    // console.log(t_str_n_last);
    // console.log(mean_pressure_value_normalized_n_last);
    // console.log(past_had_flow_n_last);

    updateChart(
      mean_pressure_value_normalized_trace_div,
      t_str_n_last,
      mean_pressure_value_normalized_n_last,
      null,
      null,
      -1
    );
    updateChart(
      past_had_flow_trace_div,
      t_str_n_last,
      past_had_flow_n_last,
      null,
      null,
      -1
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
