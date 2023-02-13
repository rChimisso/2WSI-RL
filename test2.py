import os
import sys
from datetime import datetime
import pylab as pl
from IPython import display
from matplotlib.transforms import Bbox

if 'SUMO_HOME' in os.environ:
  tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
  sys.path.append(tools)
else:
  sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

agents: dict[str, dict[str, str | bool | int]] = {
  'qlearning': {
    'color': '#0000aa',
    'fixed': False
  },
  'fixedcycle': {
    'color': '#aa0000',
    'fixed': True
  }
}
metrics: dict[str, dict[str, list | pl.Axes]] = {
  # 'step': {},
  'system_total_stopped': {},
  'system_total_waiting_time': {},
  'system_mean_waiting_time': {},
  'system_mean_speed': {},
  't_stopped': {},
  't_accumulated_waiting_time': {},
  't_average_speed': {},
  'agents_total_stopped': {},
  'agents_total_accumulated_waiting_time': {}
}
num_metrics = len(metrics)
plots_row_length = num_metrics // 2 + num_metrics % 2
plots_col_length = 2

figure = pl.figure()
figure.set_figheight(plots_row_length * 8)
figure.set_figwidth(32)
gridspec = figure.add_gridspec(plots_row_length, plots_col_length * 2)

for metric in metrics:
  metrics[metric] = { agent: [] for agent in agents }
  index = list(metrics.keys()).index(metric)
  col_index = index % 2 * 2
  metrics[metric]['plot'] = figure.add_subplot(gridspec[index // 2, col_index:(col_index + 2)])
  metrics[metric]['plot'].set_title(f'{metric} / time')
  metrics[metric]['plot'].set_xlabel('step')
  metrics[metric]['plot'].set_ylabel(metric)

def execution(name: str, color: str, fixed: bool):
  route='nets/single-intersection/single-intersection.rou.xml'
  alpha=0.1
  gamma=0.99
  epsilon=1
  min_epsilon=0.005
  decay=0.9
  min_green=10
  max_green=50
  gui=False
  seconds=100
  experiment_time = str(datetime.now()).split('.')[0]
  out_csv = f'outputs/single-intersection/{experiment_time}_alpha{alpha}_gamma{gamma}_eps{epsilon}_decay{decay}_fixed{fixed}'.replace(':', '-')

  env = SumoEnvironment(
    net_file='nets/single-intersection/single-intersection.net.xml',
    delta_time=5,
    yellow_time=3,
    route_file=route,
    out_csv_name=out_csv,
    use_gui=gui,
    num_seconds=seconds,
    min_green=min_green,
    max_green=max_green,
    fixed_ts=fixed
  )

  initial_states = env.reset()
  ql_agents = {
    ts: QLAgent(
      starting_state=env.encode(initial_states[ts], ts),
      state_space=env.observation_space,
      action_space=env.action_space,
      alpha=alpha,
      gamma=gamma,
      exploration_strategy=EpsilonGreedy(initial_epsilon=epsilon, min_epsilon=min_epsilon, decay=decay)
    ) for ts in env.ts_ids
  }

  def updateMetrics():
    for metric in metrics:
      metrics[metric][name].append(env.metrics[-1][metric])
      # metrics[metric]['plot'].plot(metrics[metric][name], color=color)
    # pl.savefig('plot.png')
    # display.clear_output(wait=True)
    # display.display(pl.gcf())
    # display.display(display.Image(filename='plot.png'))

  done = {'__all__': False}
  if fixed:
    while not done['__all__']:
      _, _, done, _ = env.step({})
      updateMetrics()
  else:
    while not done['__all__']:
      actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

      s, r, done, _ = env.step(action=actions)

      updateMetrics()

      for agent_id in ql_agents.keys():
        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

  env.save_csv(out_csv, 1)
  env.close()

  for metric in metrics:
    metrics[metric]['plot'].plot(metrics[metric][name], color=color)
    bbox = Bbox.from_extents(0, 0, 1, 1)
    pl.savefig(f'{metric}_plot.png', bbox_inches=bbox)
    # display.clear_output(wait=True)
    # display.display(pl.gcf())

  # pl.close(figure)
