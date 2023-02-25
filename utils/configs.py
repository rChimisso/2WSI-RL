from typing import Literal, TypedDict

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
  'system_mean_speed': 'Mean speed'
}

class PlotterAgentConfig(TypedDict):
  """ TypedDict for a TrafficAgent plotter configuration. """

  name: str
  """ Agent name. """
  color: str
  """ Color for the plotted lines. """

class TrafficAgentConfig(PlotterAgentConfig):
  """ TypedDict for a TrafficAgent hyperparameters configuration. """

  repeat: int
  """ How many times to repeat the run. """

class LearningAgentConfig(TrafficAgentConfig):
  """ TypedDict for a TrafficAgent using a learning model hyperparameters configuration. """

  alpha: float
  """ Learning rate. """
  gamma: float
  """ Discount rate. """
  init_eps: float
  """ Initial value for epsilon, the exploration chance. """
  min_eps: float
  """ Final value for epsilo, the exploration chance. """
  decay: float
  """
  If a QLTrafficAgent: the constant by which epsilon is multiplied to decrease it at each action taken.
  If a DQLTrafficAgent: the fraction of the training time in which to bring epsilon from init_eps to min_eps.
  """

class CanvasConfig():
  """ Configuration for a Canvas. """

  metrics: list[Metric]
  """ List of all metrics to plot (one plot for each metric). """
  plots_per_row: int
  """ How many plots draw in each canvas row. """
  dpi: int
  """ Dots Per Inch, resolution of the canvas. """

  def __init__(self, metrics: list[Metric] = ['system_total_stopped', 'system_total_waiting_time', 'system_mean_waiting_time', 'system_mean_speed'], plots_per_row: int = 1, dpi: int = 100) -> None:
    """
    Configuration for a Canvas.

    :param metrics: List of all metrics to plot (one plot for each metric).
    :type metrics: list[Metric]
    :param plots_per_row: How many plots draw in each canvas row.
    :type plots_per_row: int
    :param dpi: Dots Per Inch, resolution of the canvas.
    :type dpi: int
    """
    self.metrics = metrics
    self.plots_per_row = plots_per_row
    self.dpi = dpi
