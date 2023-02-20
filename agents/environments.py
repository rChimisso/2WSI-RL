import os
import sys
if 'SUMO_HOME' in os.environ:
  tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
  sys.path.append(tools)
else:
  sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment

class TrafficEnvironment():
  def __init__(
    self,
    net: str,
    rou: str,
    seconds: int,
    delta_time: int,
    yellow_time: int,
    min_green: int,
    max_green: int
  ) -> None:
    self.net = net
    self.rou = rou
    self.seconds = seconds
    self.delta_time = delta_time
    self.yellow_time = yellow_time
    self.min_green = min_green
    self.max_green = max_green

  def get_sumo_env(self, fixed: bool, out_csv_name: str, use_gui: bool) -> SumoEnvironment:
    return SumoEnvironment(
      net_file = self.net,
      route_file = self.rou,
      out_csv_name = out_csv_name,
      use_gui = use_gui,
      num_seconds = self.seconds,
      delta_time = self.delta_time,
      yellow_time = self.yellow_time,
      min_green = self.min_green,
      max_green = self.max_green,
      fixed_ts = fixed,
      single_agent = True
    )