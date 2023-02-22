from typing import TypedDict, Literal, Union
from pylab import Figure, GridSpec, Axes, figure
from matplotlib.transforms import Bbox
from pathlib import Path

default_plots_per_row: int = 1
default_dpi: int = 100

Metric = Literal[
  # Total number of stationary (speed < 0.1) vehicles in the current step.
  'system_total_stopped',
  # Sum of all waiting times for each vehicle. The waiting time of a vehicle is defined as the time (in seconds) spent with a speed below 0.1m/s since the last time it was faster than 0.1m/s. (basically, the waiting time of a vehicle is reset to 0 every time it moves).
  'system_total_waiting_time',
  # Arithmetic mean of all waiting times for each vehicle.
  'system_mean_waiting_time',
  # Arithmetic mean of the speed of each vehicle.
  'system_mean_speed'
]

TITLES: dict[Metric, str] = {
  'system_total_stopped': 'Number of stationary vehicles',
  'system_total_waiting_time': 'Total waiting time',
  'system_mean_waiting_time': 'Mean waiting time',
  'system_mean_speed': 'Mean speed',
  # 't_accumulated_waiting_time': 't_accumulated_waiting_time',
  # 't_average_speed': 't_average_speed',
  # 'agents_total_accumulated_waiting_time': 'agents_total_accumulated_waiting_time'
}

class PlotData():
  def __init__(self, metrics: list[Metric], plots_per_row: int = default_plots_per_row, dpi: int = default_dpi) -> None:
    self.metrics = metrics
    self.plots_per_row = plots_per_row
    self.dpi = dpi

class AgentPlotData(TypedDict):
  name: str
  color: str

class Canvas():
  def __init__(self, plot_data: PlotData) -> None:
    plots_per_col = len(plot_data.metrics) // plot_data.plots_per_row + len(plot_data.metrics) % plot_data.plots_per_row
    self._figure: Figure = figure(dpi = plot_data.dpi, figsize = (32, plots_per_col * 8))
    self._gridspec: GridSpec = self.figure.add_gridspec(plots_per_col, plot_data.plots_per_row * 2)
    self._metrics: dict[Metric, Axes] = {metric: self._get_subplot(plot_data.plots_per_row, index, metric) for index, metric in enumerate(plot_data.metrics)}

  def _get_subplot(self, plots_per_row: int, current_index: int, metric: Metric) -> Axes:
    col_index = current_index % plots_per_row * 2
    return self._init_subplot(self.figure.add_subplot(self.gridspec[current_index // plots_per_row, col_index:(col_index + 2)]), metric)

  def _init_subplot(self, plot: Axes, metric: Metric) -> Axes:
    plot.set_title(f'{TITLES[metric]} over time')
    plot.set_xlabel('Step')
    plot.set_ylabel(TITLES[metric])
    return plot

  @property
  def figure(self) -> Figure:
    return self._figure
  
  @property
  def gridspec(self) -> GridSpec:
    return self._gridspec

  @property
  def renderer(self):
    return self._figure.canvas.get_renderer()

  def get_plot(self, metric: Metric) -> Union[Axes, None]:
    if metric in self._metrics:
      return self._metrics[metric]
    return None

  def get_plots(self) -> dict[Metric, Union[Axes, None]]:
    return {metric: self.get_plot(metric) for metric in self._metrics}

  def save(self, metric: Metric, folder: Union[str, None] = None) -> None:
    plot = self.get_plot(metric)
    bbox = plot.get_tightbbox(renderer = self.renderer) if plot is not None else None
    if bbox is not None:
      dpi = self.figure.get_dpi()
      bbox = Bbox.from_extents(bbox.x0 / dpi, bbox.y0 / dpi, bbox.xmax / dpi, bbox.ymax / dpi)
      subfolder = f'{folder}/' if folder is not None else ''
      Path(f'outputs/{subfolder}plots/').mkdir(parents = True, exist_ok = True)
      self.figure.savefig(f'outputs/{subfolder}plots/{TITLES[metric]} plot.png', bbox_inches = bbox.expanded(1.01, 1.01))

class Plotter():
  def __init__(
    self,
    color: str,
    plot_data: PlotData,
    canvas: Union[Canvas, None] = None
  ) -> None:
    self.color = color
    if canvas is None:
      self.canvas = Canvas(plot_data)
    else:
      self.canvas = canvas
    self.metrics: dict[Metric, list[float]] = {
      metric: [] for metric in plot_data.metrics
    }

  def append(self, new_data: float, metric: Metric) -> None:
    if metric in self.metrics:
      self.metrics[metric].append(new_data)

  def plot(self, metric: Metric, label: Union[str, None] = None):
    if metric in self.metrics:
      plot = self.canvas.get_plot(metric)
      if plot is not None:
        if label is None:
          plot.plot(self.metrics[metric], color = self.color)
        else:
          plot.plot(self.metrics[metric], color = self.color, label = label)
          plot.legend()

  def save(self, folder: str):
    for metric in self.metrics:
      self.plot(metric)
      self.canvas.save(metric, folder)

class MultiPlotter():
  def __init__(
    self,
    agents: list[AgentPlotData],
    plot_data: PlotData
  ) -> None:
    self.metrics = plot_data.metrics
    self.canvas = Canvas(plot_data)
    self.plotters: dict[str, Plotter] = {agent['name']: Plotter(agent['color'], plot_data, self.canvas) for agent in agents}

  def append(self, new_data: float, metric: Metric, agent: str) -> None:
    if agent in self.plotters:
      self.plotters[agent].append(new_data, metric)

  def plot(self, metric: Metric, agent: str):
    if agent in self.plotters:
      return self.plotters[agent].plot(metric)

  def save(self):
    for metric in self.metrics:
      for plotter in self.plotters:
        self.plotters[plotter].plot(metric, plotter)
      self.canvas.save(metric)
