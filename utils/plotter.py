from typing import TypedDict, Literal, Union
import pylab as pl
from matplotlib.transforms import Bbox
# from matplotlib.colors 
from pathlib import Path

default_plots_per_row: int = 1
default_dpi: int = 100

Metric = Literal[
  'system_total_stopped',
  'system_total_waiting_time',
  'system_mean_waiting_time',
  'system_mean_speed',
  't_stopped',
  't_accumulated_waiting_time',
  't_average_speed',
  'agents_total_stopped',
  'agents_total_accumulated_waiting_time'
]

class AgentPlotData(TypedDict):
  name: str
  color: str

class Canvas():
  def __init__(
    self,
    metrics: list[Metric],
    plots_per_row: int = default_plots_per_row,
    dpi: int = default_dpi
  ) -> None:
    plots_per_col = len(metrics) // plots_per_row + len(metrics) % plots_per_row
    self._figure: pl.Figure = pl.figure(dpi = dpi, figsize = (32, plots_per_col * 8))
    self._gridspec: pl.GridSpec = self.figure.add_gridspec(plots_per_col, plots_per_row * 2)
    self._metrics: dict[Metric, pl.Axes] = {metric: self._get_subplot(plots_per_row, index, metric) for index, metric in enumerate(metrics)}

  def _get_subplot(self, plots_per_row: int, current_index: int, plot_name: str) -> pl.Axes:
    col_index = current_index % plots_per_row * 2
    return self._init_subplot(self.figure.add_subplot(self.gridspec[current_index // plots_per_row, col_index:(col_index + 2)]), plot_name)

  def _init_subplot(self, plot: pl.Axes, name: str) -> pl.Axes:
    plot.set_title(f'{name} over time')
    plot.set_xlabel('step')
    plot.set_ylabel(name)
    return plot

  @property
  def figure(self) -> pl.Figure:
    return self._figure
  
  @property
  def gridspec(self) -> pl.GridSpec:
    return self._gridspec

  @property
  def renderer(self):
    return self._figure.canvas.get_renderer()

  def get_plot(self, metric: Metric) -> Union[pl.Axes, None]:
    if metric in self._metrics:
      return self._metrics[metric]
    return None

  def get_plots(self) -> dict[Metric, Union[pl.Axes, None]]:
    return {metric: self.get_plot(metric) for metric in self._metrics}

  def save(self, metric: Metric, folder: Union[str, None] = None) -> None:
    plot = self.get_plot(metric)
    bbox = plot.get_tightbbox(renderer = self.renderer) if plot is not None else None
    if bbox is not None:
      dpi = self.figure.get_dpi()
      bbox = Bbox.from_extents(bbox.x0 / dpi, bbox.y0 / dpi, bbox.xmax / dpi, bbox.ymax / dpi)
      subfolder = f'{folder}/' if folder is not None else ''
      Path(f'outputs/{subfolder}plots/').mkdir(parents = True, exist_ok = True)
      self.figure.savefig(f'outputs/{subfolder}plots/{metric}_plot.png', bbox_inches = bbox.expanded(1.01, 1.01))

class Plotter():
  def __init__(
    self,
    color: str,
    metrics: list[Metric],
    plots_per_row: int = default_plots_per_row,
    dpi: int = default_dpi,
    canvas: Union[Canvas, None] = None
  ) -> None:
    self.color = color
    if canvas is None:
      self.canvas = Canvas(metrics, plots_per_row, dpi)
    else:
      self.canvas = canvas
    self.metrics: dict[Metric, list[float]] = {
      metric: [] for metric in metrics
    }

  def _get_subplot(self, plots_per_row: int, current_index: int, plot_name: str) -> pl.Axes:
    col_index = current_index % plots_per_row * 2
    return self._init_plot(self.canvas.figure.add_subplot(self.canvas.gridspec[current_index // plots_per_row, col_index:(col_index + 2)]), plot_name)

  def _init_plot(self, plot: pl.Axes, name: str) -> pl.Axes:
    plot.set_title(f'{name} over time')
    plot.set_xlabel('step')
    plot.set_ylabel(name)
    return plot

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
    metrics: list[Metric],
    plots_per_row: int = default_plots_per_row,
    dpi: int = default_dpi
  ) -> None:
    self.metrics = metrics
    self.canvas = Canvas(metrics, plots_per_row, dpi)
    self.plotters: dict[str, Plotter] = {agent['name']: Plotter(agent['color'], metrics, plots_per_row, dpi, canvas = self.canvas) for agent in agents}

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
