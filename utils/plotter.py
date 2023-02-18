from typing import TypedDict, Literal
import pylab as pl
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
from pathlib import Path

MetricName = Literal[
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

class Metric(TypedDict):
  data: list[float]
  plot: pl.Axes

class Plotter():
  def __init__(
    self,
    color: str,
    metrics: list[MetricName],
    x_lim: float,
    plots_per_row: int = 1,
    dpi: int = 100,
  ) -> None:
    self.color = color
    plots_per_col = len(metrics) // plots_per_row + len(metrics) % plots_per_row
    self.figure = pl.figure(dpi = dpi, figsize = (min(max(x_lim / 10, 32), (2**16 - 1) / dpi), plots_per_col * 8))
    self.gridspec = self.figure.add_gridspec(plots_per_col, plots_per_row * 2)
    self.metrics: dict[MetricName, Metric] = {
      metric: {
        'data': [],
        'plot': self._get_subplot(plots_per_row, index, metric)
      } for index, metric in enumerate(metrics)
    }

  def _get_subplot(self, plots_per_row: int, current_index: int, plot_name: str) -> pl.Axes:
    col_index = current_index % plots_per_row * 2
    return self._init_plot(self.figure.add_subplot(self.gridspec[current_index // plots_per_row, col_index:(col_index + 2)]), plot_name)

  def _init_plot(self, plot: pl.Axes, name: str) -> pl.Axes:
    plot.set_title(f'{name} over time')
    plot.set_xlabel('step')
    plot.set_ylabel(name)
    return plot

  def append(self, new_data: float, metric: MetricName) -> None:
    if (metric in self.metrics):
      self.metrics[metric]['data'].append(new_data)

  def plot(self, metric: MetricName):
    return self.metrics[metric]['plot'].plot(self.metrics[metric]['data'], color = self.color)

  def save(self, folder: str):
    dpi = self.figure.get_dpi()
    for metric in self.metrics:
      self.plot(metric)
      bbox = self.metrics[metric]['plot'].get_tightbbox(renderer = self.figure.canvas.get_renderer())
      if (bbox is not None):
        bbox = Bbox.from_extents(bbox.x0 / dpi, bbox.y0 / dpi, bbox.xmax / dpi, bbox.ymax / dpi)
        Path(f'outputs/plots/{folder}/').mkdir(parents = True, exist_ok = True)
        self.figure.savefig(f'outputs/plots/{folder}/{metric}_plot.png', bbox_inches = bbox.expanded(1.01, 1.01))