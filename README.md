# Study on the application of reinforcement learning to the management of a traffic light intersection.
#### Università degli Studi di Milano Bicocca, Riccardo Chimisso 866009 - Alberto Ricci 869271
---

<br/>
<br/>

# Table of Contents
- ## [Aim](#aim-1)
- ## [Reinforcement Learning](#reinforcement-learning-2)
  - ### [Reinforcement Learning](#reinforcement-learning-3)
  - ### [Q-Learning](#q-learning-1)
  - ### [Deep Reinforcement Learning](#deep-reinforcement-learning-1)
  - ### [Deep Q-Learning](#deep-q-learning-1)
  - ### [Deep Q-Network](#deep-q-network-1)
- ## [Tools](#tools-1)
  - ### [SUMO](#sumo-1)
  - ### [Sumo-RL](#sumo-rl-1)
  - ### [Matplotlib](#matplotlib-1)
  - ### [Stable Baselines 3](#stable-baselines-3-1)
  - ### [Python, Anaconda e Jupyter Notebook](#python-anaconda-e-jupyter-notebook-1)
  - ### [Visual Studio Code](#visual-studio-code-1)
  - ### [GitHub](#github-1)
- ## [Setup](#setup-1)
- ## [Codebase](#codebase-1)
- ## [Environments](#environments-1)
- ## [Experiments and results](#experiments-and-results-1)
- ## [Conclusion and possible developments](#conclusion-and-possible-developments-1)
- ## [References](#references-1)

<br/>
<br/>

# Aim

We want to compare for a particular type of traffic light intersection, referred to henceforth as **2WSI** (**2** **W**ay **S**ingle **I**ntersection), a fixed-loop traffic light management scheme with two different management schemes controlled by reinforcement learning agents.  
Specifically, 3 control schemes will then be compared:  
- Fixed-cycle: the phases of the traffic light are fixed and always repeat the same.  
- Q-Learning: the traffic light is controlled by a reinforcement learning agent using the Q-Learning technique, discussed in detail below.  
- Deep Q-Learning: the traffic light is controlled by a reinforcement learning agent using the Deep Q-Learning technique, discussed in detail below.  

Each of these models will be trained with a certain traffic situation, and then the result of the training will be tested with the same traffic situation used for training and another one that was not seen during training.  
This choice is motivated by wanting not only to compare the models with each other, but also to test how well the learning models can generalize, avoid overfitting, and thus adapt to different traffic situations.  
The robustness of the agents is very important since in reality it is easy for a traffic light intersection to be subject to variable traffic: just think of the difference between rush hour and night time, or weekday and holiday months.

<br/>
<br/>

# Reinforcement Learning

## Reinforcement Learning
Reinforcement Learning is a learning technique that involves learning of an agent through interaction with a dynamic environment. The agent interacts with the environment sequentially, performing actions and receiving a reward.  
The agent's goal is to maximize the cumulative reward that is provided by the environment in response to its actions.  
RL is based on a process of learning by exploration and experimentation, in which the agent must choose which actions to perform in order to maximize its reward. The agent learns from his experience by accumulating knowledge and developing increasingly effective strategies.  
The agent's goal is thus *max=Σ(s<sub>t</sub>, a<sub>t</sub>)* for *t = 0* to *T*, where *T* is the maximum of time steps.  
It should be noted that an agent with such goal might find itself in indecision in the case of sequences of actions whose total reward is equal. For instance, given the sequences of rewards ⧼ *0, 0, 1* ⧽ e ⧼ *1, 0, 0* ⧽ which one should the agent choose? To decide, the discount factor *γ* is introduced to decrease the weight that future rewards have over more immediate ones, so that the agent chooses the fastest maximization of cumulative reward. The discount factor is *0 ≤ γ ≤ 1* and the reward at time *t* is given by:  
*R<sub>t</sub> = r<sub>t</sub> + γr<sub>t+1</sub> + γ<sup>2</sup>r<sub>t+2</sub> + ... + γ<sup>T-t</sup>r<sub>T</sub> = Σγ<sup>i-t</sup>r<sub>i</sub> = r<sub>t</sub> + R<sub>t+1</sub>* for *i = t* to *T*, where *r<sub>i</sub>* is the reward for the time step *i*. This series is nothing but the geometric series, and as such it always converges to a finite value even for *T = ∞*.

## Q-Learning
Q-Learning is a specific Reinforcement Learning algorithm that is based on the construction of a *Q* table indicating the reward value for each possible state upon the completion of any of the possible actions.  
To construct such a table, an iterative procedure is used in which the agent explores the environment by performing more or less random actions. In detail, at each step, the table will be updated with: *Q[s][a] = Q[s][a] + ⍺(r + γ**ᐧ**max(Q[s']) - Q[s][a])*, where *s* is the current state, *a* the action performed, *r* the reward obtained, *s'* the next state, *γ* the discount factor, *⍺* the learning rate, and *max(Q[x])* returns the maximum reward obtainable from state *x*.  
This way, the table cell representing the expected value of the reward for completing action a in state s will gradually converge to the actual value.  
As anticipated, the choice of action to be performed will initially be random, until it is decided that enough has been explored. The policy often most often used for this purpose is *ε*-greedy, where given an initial *ε* representing the probability of taking a random action, we decrease that value as iterations progress to a minimum.  
QL is a very powerful and effective technique that can learn optimal action strategies in a wide range of applications. However, it can be sensitive to noise, action indeterminacy and application on continuous environments. In addition, QL requires a large amount of memory to store the *Q* table, especially when the environment has a space of possible states (*S*) and actions (*A*) (*ϴ(SA)*).

## Deep Reinforcement Learning
Deep Reinforcement Learning is a machine learning technique based on RL, but aimed at overcoming the latter's problem for very large spaces of states and actions. It does this by using deep neural networks (hence the name) to approximate the values of the *Q* table without requiring the same resources in terms of memory.

## Deep Q-Learning
The simplest approach for implementing a deep neural network as an approximator for the *Q* table is to use a deep neural network at each step to obtain the expected reward and update the neural network weights with the gradient descent method with respect to the reward actually obtained.  
However, this approach has the disadvantage of not meeting two important conditions for the likely convergence of a supervised learning method such as neural networks:  
- The targets of the neural network are not stationary, that is, they vary over time, since it is the network itself that obtains the current targets based on the predictions it makes for future targets. In fact, the neural network estimates *Q* values, which represent the expected future gain associated with a state-action pair, and these values are used to calculate the targets for updating the network's own weights. Because the neural network updates itself using its own outputs to estimate future targets, the targets are not fixed and vary continuously over time, which makes the neural network unstable and prone to nonconvergence.  
- The inputs to the neural network are not independent and identically distributed since they are generated from a time sequence with sequential correlation and depend on the *Q* table used by the agent, which may change over time as a result of experience.

## Deep Q-Network
The approach named Deep Q-Network attempts to curb the problems of the simpler DQL through the following two methods:  
- To decrease the nonstationarity of targets, a second deep neural network, called the target network, is introduced and used to estimate the targets to which the main network, must converge during training. The weights of the target network are also updated as training progresses, but with much less frequency than those of the main network. In this way, it is possible to divide the training into many small supervised learning problems that are presented to the agent sequentially. This not only increases the probability of convergence but also improves the stability of the training, although at the cost of lower convergence speed, since the most up-to-date target values are not used.  
- To reduce the impact of correlation between inputs, the Experience Replay technique is adopted, which is the use of a data structure called a replay buffer within which to save samples *(s, a, r, s')* collected by the agent during learning so that it can also train on randomly selected groups of samples from the replay buffer, which in this way allows the inputs to be made a little more i.i.d. than they actually are. In addition, this technique makes it possible to learn more from individual episodes, recall rare events, and generally make better use of the agent's accumulated experience.

<br/>
<br/>

# Tools
## SUMO
[SUMO](https://sumo.dlr.de/docs/) (Simulation of Urban MObility) is an open source urban mobility simulator.  
SUMO allows developers to simulate vehicular and pedestrian traffic in an urban environment, enabling them to test and evaluate mobility solutions such as smart traffic lights, autonomous vehicles, carpooling, and more.  
The simulator is highly customizable and allows users to define the characteristics of vehicles, roads and intersections, as well as weather and traffic conditions, to create realistic scenarios. In addition, SUMO offers several evaluation metrics, such as travel time, fuel consumption, greenhouse gas emissions, and waiting time for each vehicle, which can be used to evaluate the performance of mobility systems.  
It is used in this project as the simulator of the environment, given the 2WSI road network and the different traffic situations to be handled.  
SUMO also provides an API called [TraCI](https://sumo.dlr.de/docs/TraCI.html) (Traffic Control Interface) to enable interaction between the traffic simulator and external agents, such as intelligent traffic control systems.  
The interface provided by TraCI is socket-based, which allows users to control vehicles in the simulator, change the characteristics of the road network and traffic lights, and obtain real-time traffic status information. In addition, TraCI also allows recording and playback of traffic scenarios to analyze simulation results. TraCI is supported by a wide range of programming languages, including Python used in this project.

## Sumo-RL
[Sumo-RL](https://github.com/LucasAlegre/sumo-rl) is an open source project based on SUMO and TraCI for applying RL algorithms to traffic simulation environments for managing traffic light intersections. It provides an easy-to-use interface in Python for creating environments in SUMO and managing them using RL algorithms. In particular, a pre-made implementation of QL can be used, and it is also easily possible to integrate the provided environments with other algorithm implementations of other libraries, such as Stabe Baselines 3's DQN, as long as those implementations accept a Gymnasium environment (in case of single traffic lights) or Petting Zoo one (in case of multiple traffic lights).

## Matplotlib
[Matplotlib](https://matplotlib.org/) is a Python library for creating static, animated or interactive visualizations. Because it is simple to use and already integrated with Jupyter Notebook, it is used in this project to create and save graphs of various metrics collected during the execution of RL algorithms.  
Initially, it was also considered to allow real-time visualization of the construction of the metrics plots, but although this was achieved, it was chosen to exclude this feature since it was not worth the enormous slowdown that fell on the simulations.

## Stable Baselines 3
[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) is another open source Python library that provides implementations of multiple RL and DRL algorithms, of which the one for DQN was selected in this project, and integration with Gymnasium and Petting Zoo environments. It was actually necessary to install a particular version of SB3 to ensure effective integration with Gymnasium environments.

## Python, Anaconda e Jupyter Notebook
Python was the programming language chosen for this project given its great suitability in machine learning applications, and because both Sumo-RL and Matplotlib are written and easily integrated in Python. In addition, this allowed the use of Jupyter Notebook, which facilitates the execution and sharing of the project's main script, and also allows the display of graphs in real time (as mentioned however, this has been removed) and the display of completed graphs (also a feature actually removed to avoid overly long outputs, but it can be easily reactivated).  
The version of Python used is 3.9.13 via [Anaconda](https://www.anaconda.com/products/distribution), since SB3 uses [Pytorch](https://pytorch.org/) which in turn requires a version no higher than 3.9 of Python. Also through Anaconda and Pytorch it was possible to run neural networks directly on the NVIDIA GPUs at our disposal.

## Visual Studio Code
[Visual Studio Code](https://code.visualstudio.com/) was the IDE chosen for developing the project code, from xml files for building the SUMO environments, to Python and Jupyter Notebook files for writing, implementing, and executing the experiments. This editor was chosen because it was already known and had very easy integration of Git, Jupyter Notebook, Python and GitHub.

## GitHub
Lastly, [GitHub](https://github.com/) was the last of the main tools used in the implementation of this project, allowing easy coordination between parallel developments (this more Git than GitHub) and a space to save, publish, and share the project repository while still maintaining exclusive ownership of it. It also made it easy to choose a license for the repository, as well as offered the ability to properly view the README in Markdown which contains more inherent and detailed instructions and explanations of the setup, explanation, and implementation of the project code compared to the PDF relation.  
Although the README contains and integrates the relation itself, the relation in PDF was still uploaded to the repository to allow easier viewing of it, with the advantage of better viewing the table of contents, the division between pages (and thus topics), and Latex formulas.  
Also, the main difference between the report and the README is that the latter is written in English, while the relation is written in Italian.

<br/>
<br/>

# Setup
To setup the project the following steps are needed:  
- Install Python via Anaconda following the steps [here](https://www.anaconda.com/products/distribution).  
- Install SUMO following the guide [here](https://sumo.dlr.de/docs/Downloads.php).  
- Install Pytorch via Conda with the following command:  
  `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`  
  This will install all required Pytorch modules along with the support to run Pytorch directly on your NVIDIA GPU.  
- Install the latest version of Sumo-RL via `pip3 install sumo-rl`  
- Install the latest version of Matplotlib via `pip3 install matplotlib`  
- Install Stable Baselines 3 via `pip3 install git+https://github.com/DLR-RM/stable-baselines3@feat/gymnasium-support`  
  This, unlike simply downloading the latest version, will install the unreleased version of SB3 with support for Gymnasium environments.  
  This is required because Sumo-RL uses Gymnasium internally, but the latest version of SB3 still uses Gym. You can read more [here](https://github.com/DLR-RM/stable-baselines3/pull/780#issue-1144872152).

If you're using VSCode, after these simple steps everything is ready to go!

<br/>
<br/>

# Codebase


<br/>
<br/>

# Environments

<br/>
<br/>

# Experiments and results

<br/>
<br/>

# Conclusion and possible developments

<br/>
<br/>

# References
- P. Alvarez Lopez, M. Behrisch, L. Bieker-Walz, J. Erdmann, Yun-Pang Flötteröd, R. Hilbrich, L. Lücken, J. Rummel, P. Wagner, E. Wiessner, (2018).  
  Microscopic Traffic Simulation using SUMO.  
  IEEE Intelligent Transportation Systems Conference (ITSC).  
  https://elib.dlr.de/124092/  
- Lucas N. Alegre, (2019).  
  SUMO-RL.  
  GitHub repository.  
  https://github.com/LucasAlegre/sumo-rl  
- A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann, (2021).  
  Stable-Baselines3: Reliable Reinforcement Learning Implementations.  
  Journal of Machine Learning Research.  
  http://jmlr.org/papers/v22/20-1364.html  
