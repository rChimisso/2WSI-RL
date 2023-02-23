from pathlib import Path
from typing import Union
from pylab import Figure, GridSpec, Axes, figure
from matplotlib.transforms import Bbox
from utils.configs import Metric, CanvasConfig, PlotterAgentConfig

TITLES: dict[Metric, str] = {
  'system_total_stopped': 'Number of stationary vehicles',
  'system_total_waiting_time': 'Total waiting time',
  'system_mean_waiting_time': 'Mean waiting time',
  'system_mean_speed': 'Mean speed'
}

class Canvas():
  """ Canvas to plot metrics. """

  def __init__(self, config: CanvasConfig) -> None:
    """
    Canvas to plot metrics.

    :param config: (CanvasConfig) Canvas configuration.
    """
    plots_per_col = len(config.metrics) // config.plots_per_row + len(config.metrics) % config.plots_per_row
    self._figure: Figure = figure(dpi = config.dpi, figsize = (48, plots_per_col * 8))
    self._gridspec: GridSpec = self.figure.add_gridspec(plots_per_col, config.plots_per_row * 2)
    self._metrics: dict[Metric, Axes] = {metric: self._get_subplot(config.plots_per_row, index, metric) for index, metric in enumerate(config.metrics)}

  def _get_subplot(self, plots_per_row: int, index: int, metric: Metric) -> Axes:
    """
    Returns a new subplot to add to the canvas figure.

    :param plots_per_row: (int) How many plots draw in each canvas row.
    :param index: (int) Index of the metric in the list of all metrics to plot.
    :param metric: (Metric) Metric to plot.

    :return: The subplot.
    """
    col_index = index % plots_per_row * 2
    return self._init_subplot(self.figure.add_subplot(self.gridspec[index // plots_per_row, col_index:(col_index + 2)]), metric)

  def _init_subplot(self, plot: Axes, metric: Metric) -> Axes:
    """
    Initializes and returns the given subplot.

    :param plot: (Axes) Subplot to initialize.
    :param metric: (Metric) Metric to plot.

    :return: The subplot.
    """
    plot.set_title(f'{TITLES[metric]} over time')
    plot.set_xlabel('Step')
    plot.set_ylabel(TITLES[metric])
    return plot

  @property
  def figure(self) -> Figure:
    """ Canvas figure. """
    return self._figure

  @property
  def gridspec(self) -> GridSpec:
    """ Canvas GridSpec. """
    return self._gridspec

  @property
  def renderer(self):
    """ Canvas Renderer. """
    return self._figure.canvas.get_renderer()

  def plot(self, metric: Metric, data: list[float], color: str, label: Union[str, None] = None, width: int = 1) -> None:
    """
    Plots the given data in the plot of the given metric, creating a line with the given color and optionally the associated label.

    :param metric: (Metric) Metric for which plot the data.
    :param data: (list[float]) Data to plot.
    :param color: (str) Color of the line.
    :param label: (Union[str, None]) Optional label for the line.
    :param width: (int) Optional line width.
    """
    plot = self.get_plot(metric)
    if plot is not None:
      if label is None:
        plot.plot(data, color = color, linewidth = width)
      else:
        plot.plot(data, color = color, label = label, linewidth = width)
        plot.legend()

  def get_plot(self, metric: Metric) -> Union[Axes, None]:
    """
    Returns the plot for the given metric.

    :param metric: (Metric) Metric of the plot.

    :return: The plot for the given metric or, if no plot was saved for the given metric, None.
    """
    if metric in self._metrics:
      return self._metrics[metric]
    return None

  def save(self, metric: Metric, folder: Union[str, None] = None) -> None:
    """
    Saves the plot of the given metrics, if any.
    
    :param metric: (Metric) Metric of the plot.
    :param folder: (Union[str, None]) Subfolder in which to save the plots.
    """
    plot = self.get_plot(metric)
    bbox = plot.get_tightbbox(renderer = self.renderer) if plot is not None else None
    if bbox is not None:
      dpi = self.figure.get_dpi()
      bbox = Bbox.from_extents(bbox.x0 / dpi, bbox.y0 / dpi, bbox.xmax / dpi, bbox.ymax / dpi)
      subfolder = f'{folder}/' if folder is not None else ''
      Path(f'outputs/{subfolder}plots/').mkdir(parents = True, exist_ok = True)
      self.figure.savefig(f'outputs/{subfolder}plots/{TITLES[metric]} plot.png', bbox_inches = bbox.expanded(1.01, 1.01))

  def clear(self) -> None:
    """ Clears all plots. """
    for plot in self._metrics.values():
      plot.clear()

class Plotter():
  """ Plotter for several runs and metrics of a single TrafficAgent. """

  def __init__(self, color: str, canvas_config: CanvasConfig, canvas: Union[Canvas, None] = None) -> None:
    """
    Plotter for several runs and metrics of a single TrafficAgent.
    
    :param color: (str) Color of the mean.
    :param canvas_config: (CanvasConfig) Canvas configuration, used to instantiate a new Canvas, if canvas is None, and retrieve the list of metrics to plot.
    :param canvas: (Union[Canvas, None]) Optional Canvas instance to use to plot.
    """
    self.color: str = color
    if canvas is None:
      self._canvas: Canvas = Canvas(canvas_config)
    else:
      self._canvas = canvas
    self._init_metrics(canvas_config.metrics)

  @property
  def metrics(self) -> list[Metric]:
    """ Metrics that will be plotted. """
    return [metric for metric in self._means]

  @property
  def means(self) -> dict[Metric, list[float]]:
    """ Data of the means, defined as the arithmetic mean of each metric of each run. """
    return self._means

  def add_run(self, data: dict[Metric, list[float]]) -> None:
    """
    Adds the data of a run to the list of data to plot.

    :param data: (dict[Metric, list[float]]) Data of the run.
    """
    if data:
      self._runs.append({metric: data[metric] for metric in self.metrics})

  def plot(self, metric: Metric, label: Union[str, None] = None, only_mean: bool = False) -> None:
    """
    Plots the graph for the given metric, optionally using the provided label.

    :param metric: (Metric) Metric for which plot the data.
    :param label: (Union[str, None]) Optional label that the mean line will have associated.
    """
    self._means[metric] = []
    for run in self._runs:
      if not only_mean:
        self._canvas.plot(metric, run[metric], f'{self.color}7f')
      for step, value in enumerate(run[metric]):
        if len(self._means[metric]) > step:
          self._means[metric][step] += value / len(self._runs)
        else:
          self._means[metric].append(value / len(self._runs))
    self._canvas.plot(metric, self._means[metric], self.color, label, 2)

  def save(self, folder: str) -> None:
    """
    Plots and then saves the graphs for each metric.

    :param folder: (str) Subfolder in which to save the plots.
    """
    for metric in self.metrics:
      self.plot(metric)
      self._canvas.save(metric, folder)

  def clear(self) -> None:
    """ Clears all plots and empties all data. """
    self._init_metrics(list(self._means.keys()))
    self._canvas.clear()
  
  def _init_metrics(self, metrics: list[Metric]) -> None:
    """
    Initializes the given metrics.

    :param metrics: List of metrics to initialize.
    :type metrics: list[Metric]
    """
    self._means: dict[Metric, list[float]] = {metric: [] for metric in metrics}
    self._runs: list[dict[Metric, list[float]]] = []

class MultiPlotter():
  """ Plotter for several TrafficAgents. """

  def __init__(self, agents: list[PlotterAgentConfig], canvas_config: CanvasConfig) -> None:
    """ 
    Plotter for several TrafficAgents.

    :param agents: List of configuration for each TrafficAgent.
    :type agents: list[PlotterAgentConfig]
    :param canvas_config: Canvas configuration.
    :type canvas_config: CanvasConfig
    """
    self.metrics = canvas_config.metrics
    self.canvas = Canvas(canvas_config)
    self.plotters: dict[str, Plotter] = {agent['name']: Plotter(agent['color'], canvas_config, self.canvas) for agent in agents}

  def add_run(self, data: dict[Metric, list[float]], agent: str) -> None:
    """
    Adds the data of a run of an agent to the list of data to plot.

    :param data: Data of the run.
    :type data: dict[Metric, list[float]]
    :param agent: Name of the agent that did the run.
    :type agent: str
    """
    if agent in self.plotters:
      self.plotters[agent].add_run(data)

  def plot(self, metric: Metric, agent: str) -> None:
    """
    Plots and then saves the graphs for each metric.

    :param metric: (Metric) Metric for which plot the data.
    :type metric: Metric
    :param agent: Name of the agent that produced the data.
    :type agent: str
    """
    if agent in self.plotters:
      self.plotters[agent].plot(metric)

  def save(self) -> None:
    """ Plots and then saves the graphs for each metric. """
    for metric in self.metrics:
      for plotter in self.plotters:
        self.plotters[plotter].plot(metric, plotter, True)
      self.canvas.save(metric)
  
  def clear(self) -> None:
    """ Clears all plots and empties all data. """
    for plotter in self.plotters.values():
      plotter.clear()
