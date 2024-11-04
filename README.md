# Reinforcement Learning Lecture Notes: From Basics to Advanced

## Part 1: Understanding the Basics

### What is Reinforcement Learning?
Think of training a pet dog. When the dog does something good, you give it a treat. When it does something wrong, you don't. Over time, the dog learns what actions lead to treats. This is exactly how reinforcement learning works in AI! The AI (like our dog) learns by trying things out and getting rewards for good actions.

### Real-World Examples:
1. Video game AI learning to win games
2. Robot learning to walk
3. Self-driving cars learning to navigate
4. Chess programs learning winning strategies

### Basic Components:
1. Agent = The learner (like our dog)
2. Environment = The world around the agent
3. State = Current situation
4. Action = What the agent can do
5. Reward = Feedback for actions

## Part 2: Mathematical Foundation

### Basic Notation:
- State: $s_t$ (at time t)
- Action: $a_t$
- Reward: $r_t$
- Policy: $\pi$ (strategy for choosing actions)

### The Learning Process:
1. Agent observes state $s_t$
2. Takes action $a_t$
3. Gets reward $r_t$
4. Moves to new state $s_{t+1}$

### Value Functions
1. State-Value Function (how good is a state):
$V_\pi(s) = E_\pi[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1}|s_t=s]$

2. Action-Value Function (how good is an action in a state):
$Q_\pi(s,a) = E_\pi[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1}|s_t=s,a_t=a]$

## Part 3: Core Concepts

### Markov Decision Process (MDP)
- Formal way to describe the RL problem
- Components: $(S, A, P, R, \gamma)$
  - S: Set of states
  - A: Set of actions
  - P: Transition probability
  - R: Reward function
  - γ: Discount factor

### Bellman Equation
The fundamental equation:
$V_\pi(s) = \sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)[r + \gamma V_\pi(s')]$

## Part 4: Learning Algorithms

### 1. Q-Learning
Basic update rule:
$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_t + \gamma \max_a Q(s_{t+1},a) - Q(s_t,a_t)]$

### 2. SARSA
Update rule:
$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_t + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)]$

## Part 5: Advanced Topics

### Deep Reinforcement Learning
Combining neural networks with RL:
1. DQN (Deep Q-Network)
2. Policy Gradients:
   $\nabla J(\theta) = E_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)Q^{\pi_\theta}(s,a)]$

### Actor-Critic Methods
Combines:
- Actor: Learns policy $\pi_\theta(a|s)$
- Critic: Learns value function $V_w(s)$

### Practical Applications:

1. **Game Playing**
   - State: Game position
   - Actions: Possible moves
   - Reward: Win/Lose/Points

2. **Robotics**
   - State: Joint positions, sensor readings
   - Actions: Motor commands
   - Reward: Task completion metrics

## Part 6: Implementation Considerations

### Common Challenges:
1. Exploration vs Exploitation
2. Credit Assignment
3. Sample Efficiency
4. Stability

### Best Practices:
1. Start simple
2. Use appropriate reward design
3. Consider environment complexity
4. Handle continuous spaces carefully



––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Value-Based Methods

![Trained Agents][image1]

This repository contains material related to Udacity's Value-based Methods course.

## Table of Contents

### Tutorials

The tutorials lead you through implementing various algorithms in reinforcement learning.  All of the code is in PyTorch (v0.4) and Python 3.

* [Deep Q-Network](https://github.com/udacity/Value-based-methods/tree/main/dqn): Explore how to use a Deep Q-Network (DQN) to navigate a space vehicle without crashing.

### Labs / Projects

The labs and projects can be found below.  All of the projects use rich simulation environments from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents).

* [Navigation](https://github.com/udacity/Value-based-methods/tree/main/p1_navigation): In the first project, you will train an agent to collect yellow bananas while avoiding blue bananas.

### Resources

* [Cheatsheet](https://github.com/udacity/Value-based-methods/tree/main/cheatsheet): You are encouraged to use [this PDF file](https://github.com/udacity/Value-based-methods/blob/main/cheatsheet/cheatsheet.pdf) to guide your study of reinforcement learning. 

## OpenAI Gym Benchmarks

### Box2d
- `LunarLander-v2` with [Deep Q-Networks (DQN)](https://github.com/udacity/Value-based-methods/blob/main/dqn/solution/Deep_Q_Network_Solution.ipynb) | solved in 1504 episodes

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/Value-based-methods.git
cd Value-based-methods/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

## Want to learn more?

<p align="center">Come learn with us in the <a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">Deep Reinforcement Learning Nanodegree</a> program at Udacity!</p>

<p align="center"><a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">
 <img width="503" height="133" src="https://user-images.githubusercontent.com/10624937/42135812-1829637e-7d16-11e8-9aa1-88056f23f51e.png"></a>
</p>
