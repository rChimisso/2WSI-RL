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
  """ TypedDict for a LearningTrafficAgent hyperparameters configuration. """

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
  If a QLearningTrafficAgent: the constant by which epsilon is multiplied to decrease it at each action taken.
  If a DeepQLearningTrafficAgent: the fraction of the training time in which to bring epsilon from init_eps to min_eps.
  """

class CanvasConfig():
  """ Configuration for a Canvas. """

  metrics: list[Metric]
  """ List of all metrics to plot (one plot for each metric). """
  plots_per_row: int
  """ How many plots draw in each canvas row. """
  dpi: int
  """ Dots Per Inch, resolution of the canvas. """

  def __init__(self, metrics: list[Metric], plots_per_row: int = 1, dpi: int = 100) -> None:
    """
    Configuration for a Canvas.

    :param metrics: (list[Metric]) List of all metrics to plot (one plot for each metric).
    :param plots_per_row: (int) How many plots draw in each canvas row.
    :param dpi: (int) Dots Per Inch, resolution of the canvas.
    """
    self.metrics = metrics
    self.plots_per_row = plots_per_row
    self.dpi = dpi
