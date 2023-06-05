// get div by ID
let datetime_box_div = document.getElementById("datetime_box");
let pressure_value_box_div = document.getElementById("pressure_value_box");
let had_flow_box_div = document.getElementById("had_flow_box");

let pressure_value_normalized_gauge_div = document.getElementById("pressure_value_normalized_gauge");

let mean_div = document.getElementById("mean_div");
let live_div = document.getElementById("live_div");

// colors
const c0 = "#012169";
const c1 = "#993399";
const c_grey = "#7f7f7f";

// markers
const ms_flow_0 = "bowtie";
const ms_flow_1 = "bowtie-open";
const mc_flow_0 = c0;
const mc_flow_1 = c1;
const marker_size_large = 12;
const marker_size_small = 6;

// flow null traces
const legend_trace_flow_0 = {
  x: [null],
  y: [null],
  name: "No Flow",
  type: "scatter",
  mode: "markers",
  marker: {
    size: marker_size_large,
    line: {width: 1.5},
    symbol: ms_flow_0,
    color: mc_flow_0,
  },
};
const legend_trace_flow_1 = {
  x: [null],
  y: [null],
  name: "Had Flow",
  type: "scatter",
  mode: "markers",
  marker: {
    size: marker_size_large,
    line: {width: 1.5},
    symbol: ms_flow_1,
    color: mc_flow_1,
  },
};

// config for all graphs
const graph_config = {
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
let pressure_value_normalized_gauge_data = [
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

let gauge_layout = { width: 400, height: 100, margin: { t: 0, b: 0, l: 0, r: 0 } };

Plotly.newPlot(pressure_value_normalized_gauge_div, pressure_value_normalized_gauge_data, gauge_layout, graph_config);

function updateGauge(pressure_value_normalized) {
  let pressure_value_normalized_update = {
    value: pressure_value_normalized,
  };

  Plotly.update(pressure_value_normalized_gauge_div, pressure_value_normalized_update);
}

// trace layout
let mean_layout = {
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
  shapes: [{
    type: 'line',
    xref: 'paper',
    x0: 0,
    x1: 1,
    yref: 'y',
    y0: 100,
    y1: 100,
    line: {
      color: c_grey,
      dash: "dash",
    },
  }],
  autosize: true,
  margin: {t: 0, b: 70, l: 70, r: 0},
  font: {color: c_grey, size: 14},
};

let mean_trace = {
  x: [],
  y: [],
  type: "scatter",
  mode: "lines+markers",
  marker: {
    size: marker_size_large,
    line: {width: 1.5},
    symbol: [],
  },
  showlegend: false, 
};

let live_layout = JSON.parse(JSON.stringify(mean_layout))
let live_trace = JSON.parse(JSON.stringify(mean_trace))

live_layout["xaxis"]["title"] = "Polling Samples";
live_layout["xaxis"]["range"] = [0,60];
live_layout["yaxis"]["title"] = "Pressure %";
live_layout["showlegend"] = false;
live_trace["mode"] = "markers";
live_trace["marker"]["size"] = marker_size_small;

Plotly.newPlot(
  mean_div,
  [mean_trace, legend_trace_flow_0, legend_trace_flow_1],
  mean_layout,
  graph_config
);

Plotly.newPlot(
  live_div,
  [live_trace, legend_trace_flow_0, legend_trace_flow_1],
  live_layout,
  graph_config
);

let live_x_array = [];
let live_y_array = [];
let live_ms_array = [];
let live_mc_array = [];

function update_chart(div, x_array, y_array, ms_array, mc_array, max_graph_points, marker_size=marker_size_large, x_value=null, y_value=null, z_value=null) {
  if ( 0 < max_graph_points) {
    if (max_graph_points <= x_array.length ) {
      x_array.shift();
      y_array.shift();
      ms_array.shift();
      mc_array.shift();
    }
    x_array.push(x_value);
    y_array.push(y_value);

    if (z_value == 0) {
      ms_array.push(ms_flow_0);
      mc_array.push(mc_flow_0);
    }
    else {
      ms_array.push(ms_flow_1);
      mc_array.push(mc_flow_1);
    }

  }

  let data_update = {
    x: [x_array],
    y: [y_array],
    marker: {
      size: marker_size,
      line: {width: 1.5},
      symbol: ms_array,
      color: mc_array,
    },
  };
  // console.log(data_update);

  Plotly.update(div, data_update, {}, [0]);
}

// update everything each msg
function updateAll(data) {
  let t_str = data.t_est_str;
  let i_polling = parseInt(data.i_polling);
  let pressure_value = parseInt(data.pressure_value);
  let pressure_value_normalized = (100*parseFloat(data.pressure_value_normalized)).toFixed(2);
  let had_flow = parseInt(data.had_flow);

  updateBoxes(t_str, i_polling, pressure_value, had_flow);

  updateGauge(pressure_value_normalized);

  let updated_mean = false;
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
    let ms_n_last = []
    let mc_n_last = []

    for (let i = 0; i < mean_pressure_value_normalized_n_last.length; i++) {
      mean_pressure_value_normalized_n_last[i] = (100*parseFloat(mean_pressure_value_normalized_n_last[i])).toFixed(2);
      past_had_flow_n_last[i] = parseInt(past_had_flow_n_last[i]);

      if (past_had_flow_n_last[i] == 0) {
        ms_n_last[i] = ms_flow_0;
        mc_n_last[i] = mc_flow_0;
      }
      else {
        ms_n_last[i] = ms_flow_1;
        mc_n_last[i] = mc_flow_1;
      }
    }

    update_chart(
      mean_div,
      t_str_n_last,
      mean_pressure_value_normalized_n_last,
      ms_n_last,
      mc_n_last,
      -1,
    );
  }

  update_chart(
    live_div,
    live_x_array,
    live_y_array,
    live_ms_array,
    live_mc_array,
    60,
    marker_size_small,
    i_polling,
    pressure_value_normalized,
    had_flow
  );

}

// SocketIO Code
// let socket = io.connect("http://" + document.domain + ":" + location.port);

let socket = io.connect();

// receive data from server
socket.on("emit_data", function (msg) {
  let data = JSON.parse(msg);
  // write data to console for debugging
  // console.log(msg);
  updateAll(data);
});
