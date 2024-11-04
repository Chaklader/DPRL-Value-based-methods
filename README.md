# Reinforcement Learning Lecture Notes: From Basics to Advanced

## Part 1: Understanding the Basics

### What is Reinforcement Learning?

Think of training a pet dog. When the dog does something good, you give it a treat. When it does something wrong, you
don't. Over time, the dog learns what actions lead to treats. This is exactly how reinforcement learning works in AI!
The AI (like our dog) learns by trying things out and getting rewards for good actions.

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

<br>

![localImage](images/reward.png)

<br>


A reward function for training a robot to walk using Reinforcement Learning. Let me break down each
component:

The overall reward function is:
$r = min(v_x, v_{max}) - 0.005(v_y^2 + v_z^2) - 0.05y^2 - 0.02||u||^2 + 0.02$

There are 4 main desired behaviors being encouraged:

1. **Walk Fast**

- Represented by $min(v_x, v_{max})$
- Rewards the robot for moving forward (x-direction) up to a maximum velocity
- "Proportional to the robot's forward velocity"

2. **Walk Forward**

- Penalizes sideways and vertical motion with $-0.005(v_y^2 + v_z^2)$
- Penalizes deviation from center with $-0.05y^2$
- Wants robot to walk straight ahead without swaying or bouncing

3. **Walk Smoothly**

- The term $-0.02||u||^2$ penalizes large torques
- Encourages smooth, efficient movements
- Discourages jerky or unstable motions

4. **Walk for as long as possible**

- The constant term +0.02 provides a small positive reward for each timestep
- Encourages the robot to maintain balance and keep walking
- "Constant reward for not falling"

This is a well-designed reward function because it:

- Has clear objectives
- Balances multiple competing goals
- Uses negative penalties to discourage unwanted behaviors
- Includes both instantaneous feedback (velocity) and long-term goals (staying upright)
- Has carefully tuned coefficients to weight different objectives appropriately

The design demonstrates key principles of reward shaping in RL, where you need to carefully specify what behaviors you
want while avoiding unintended consequences in the learning process.

# Returns in Reinforcement Learning

## 1. Basic Understanding

### What is a Return?

A return is the total reward an agent receives over time. Think of it like:

- Playing a video game and adding up all points you get
- Investing money and calculating total profits over years
- A robot learning to walk and summing up all rewards from start to finish

## 2. Types of Returns

### 2.1 Cumulative Return (Simple Sum)

The simple sum of all rewards:

$G_t = R_{t+1} + R_{t+2} + R_{t+3} + ... + R_T$

Where:

- $G_t$ is the return at time t
- $R_t$ is the reward at time t
- T is the final time step

**Example:**
If rewards are [1, 2, 3, 4]:
$G_1 = 1 + 2 + 3 + 4 = 10$

### 2.2 Discounted Return

Future rewards are worth less than immediate rewards:

$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

Where:

- $\gamma$ is the discount factor (0 ≤ γ ≤ 1)
- $\gamma^k$ reduces future rewards

**Example:**
With γ = 0.9 and rewards [1, 2, 3, 4]:
$G_1 = 1 + (0.9 × 2) + (0.9^2 × 3) + (0.9^3 × 4)$
$G_1 = 1 + 1.8 + 2.43 + 2.916 = 8.146$

## 3. Why Use Discounted Returns?

### Advantages:

1. **Uncertainty**: Future rewards are less certain
2. **Immediate Focus**: Encourages quick solutions
3. **Mathematical Convenience**: Helps with infinite horizons
4. **Real-world Similarity**: Models real economic decisions

### Different γ Values:

- γ = 0: Only cares about immediate reward
- γ = 1: All rewards equally important
- γ = 0.9: Common choice balancing present and future

## 4. Practical Applications

### Example: Robot Navigation

```textmate
Immediate reward (reach goal): +10
Each step penalty: -1

Without discount (γ = 1):
- Long path (10 steps): 10 - 10 = 0
- Short path (5 steps): 10 - 5 = 5

With discount (γ = 0.9):
- Long path: 10γ¹⁰ - (1 + γ + γ² + ... + γ⁹)
- Short path: 10γ⁵ - (1 + γ + γ² + γ³ + γ⁴)
```

## 5. Mathematical Properties

### 5.1 Finite Horizon

When there's a clear end time T:
$G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k+1}$

### 5.2 Infinite Horizon

When there's no clear end:
$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

### 5.3 Recursive Relationship

Important property:
$G_t = R_{t+1} + \gamma G_{t+1}$

## 6. Implementation Considerations

### Key Points:

1. Choose γ based on task requirements
2. Consider time horizon (finite vs infinite)
3. Balance immediate vs future rewards
4. Account for computational limitations

# Lecture Notes: Markov Decision Process (MDP)

## 1. Basic Understanding

### What is MDP?

An MDP is a mathematical framework for modeling decision-making in situations where outcomes are partly random and
partly under the control of a decision-maker. It's named after Andrey Markov.

### Key Property: The Markov Property

"The future depends only on the present, not on the past"

Mathematically:
$P(s_{t+1}|s_t, a_t) = P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1},...,s_0, a_0)$

## 2. Components of MDP

### Formal Definition:

An MDP is defined by a tuple $(S, A, P, R, \gamma)$ where:

1. **States (S)**:
    - Set of all possible states
    - Example: Positions in a game, robot configurations

2. **Actions (A)**:
    - Set of all possible actions
    - Example: Move left/right, apply force

3. **Transition Probability (P)**:
    - $P(s'|s,a)$ = probability of reaching state s' from state s with action a
    - Mathematically: $P: S × A × S → [0,1]$

4. **Reward Function (R)**:
    - $R(s,a,s')$ = immediate reward for transition from s to s' with action a
    - Mathematically: $R: S × A × S → \mathbb{R}$

5. **Discount Factor (γ)**:
    - $\gamma \in [0,1]$
    - Balances immediate vs future rewards

## 3. Decision Making in MDPs

### 3.1 Policy

- A policy π defines behavior of the agent
- Can be:
    - Deterministic: $\pi(s) → a$
    - Stochastic: $\pi(a|s)$ = probability of taking action a in state s

### 3.2 Value Functions

**State-Value Function:**
$V_\pi(s) = E_\pi[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}|S_t=s]$

**Action-Value Function:**
$Q_\pi(s,a) = E_\pi[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}|S_t=s,A_t=a]$

## 4. Solving MDPs

### 4.1 Bellman Equations

**Bellman Expectation Equation:**
$V_\pi(s) = \sum_a \pi(a|s)\sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_\pi(s')]$

**Bellman Optimality Equation:**
$V_*(s) = \max_a\sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_*(s')]$

### 4.2 Solution Methods

1. **Dynamic Programming**
    - Value Iteration
    - Policy Iteration

2. **Model-Free Methods**
    - Q-Learning
    - SARSA

## 5. Example: Grid World MDP

```textmate
States: Grid cells
Actions: Up, Down, Left, Right
Transitions: 80% intended direction, 20% random
Rewards: -1 per step, +10 goal, -10 trap
```

## 6. Practical Considerations

### 6.1 Challenges

1. State Space Size
2. Partial Observability
3. Continuous States/Actions
4. Model Uncertainty

### 6.2 Applications

1. Robotics
2. Game AI
3. Resource Management
4. Healthcare Decisions

## 7. Extensions

### 7.1 Partially Observable MDPs (POMDPs)

- Agent cannot directly observe state
- Must maintain belief state

### 7.2 Continuous MDPs

- Infinite state/action spaces
- Requires function approximation

## 8. Key Takeaways

1. MDPs provide framework for sequential decision making
2. Markov property simplifies analysis
3. Solutions balance immediate vs future rewards
4. Various methods exist for finding optimal policies

# Notes:

- In general, the state space 𝒮 is the set of all nonterminal states.
- In continuing tasks (like the recycling task detailed in the video), this is equivalent to the set of all states.
- In episodic tasks, we use 𝒮⁺ to refer to the set of all states, including terminal states.
- The action space 𝒜 is the set of possible actions available to the agent.
- In the event that there are some states where only a subset of the actions are available, we can also use 𝒜(s) to
  refer to the set of actions available in state s ∈ 𝒮.

### One-Step Dynamics: State transitions and rewards for robot battery states with actions

<br>

![localImage](images/robot.png)

<br>


At an arbitrary time step t, the agent-environment interaction has evolved as a sequence of states, actions, and rewards

$(S_0, A_0, R_1, S_1, A_1, ..., R_{t-1}, S_{t-1}, A_{t-1}, R_t, S_t, A_t)$.

When the environment responds to the agent at time step t + 1, it considers only the state and action at the previous
time step $(S_t, A_t)$.

In particular, it does not care what state was presented to the agent more than one step prior. (In other words, the
environment does not consider any of $S_0,...,S_{t-1}$.)

And, it does not look at the actions that the agent took prior to the last one. (In other words, the environment does
not consider any of $A_0,...,A_{t-1}$.)

Furthermore, how well the agent is doing, or how much reward it is collecting, has no effect on how the environment
chooses to respond to the agent. (In other words, the environment does not consider any of $R_0,...,R_t$.)

Because of this, we can completely define how the environment decides the state and reward by specifying

$p(s',r|s,a) \doteq \mathbb{P}(S_{t+1} = s', R_{t+1} = r|S_t = s, A_t = a)$

for each possible s', r, s, and a. These conditional probabilities are said to specify the one-step dynamics of the
environment.


<br>

![localImage](images/summary.png)

<br>

# The Setting, Revisited

• The reinforcement learning (RL) framework is characterized by an agent learning to interact with its environment.
• At each time step, the agent receives the environment's state (the environment presents a situation to the agent), and
the agent must choose an appropriate action in response. One time step later, the agent receives a reward (the
environment indicates whether the agent has responded appropriately to the state) and a new state.
• All agents have the goal to maximize expected cumulative reward, or the expected sum of rewards attained over all time
steps.

# Episodic vs. Continuing Tasks

• A task is an instance of the reinforcement learning (RL) problem.
• Continuing tasks are tasks that continue forever, without end.
• Episodic tasks are tasks with a well-defined starting and ending point.
- In this case, we refer to a complete sequence of interaction, from start to finish, as an episode.
• Episodic tasks come to an end whenever the agent reaches a terminal state.

# The Reward Hypothesis

• Reward Hypothesis: all goals can be framed as the maximization of (expected) cumulative reward.

# Goals and Rewards

(Please see Part 1 and Part 2 to review an example of how to specify the reward signal in a real-world problem.)

# Cumulative Reward

• The return at time step t is $G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...$
• The agent selects actions with the goal of maximizing expected (discounted) return.
(Note: discounting is covered in the next concept.)

# Discounted Return

• The discounted return at time step t is $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$
• The discount rate γ is something that you set to refine the goal that you have for the agent.
• It must satisfy 0 ≤ γ ≤ 1.
• If γ = 0, the agent only cares about the most immediate reward.
• If γ = 1, the return is not discounted.
• For larger values of γ, the agent cares more about the distant future. Smaller values of γ result in more extreme
discounting, where - in the most extreme case - agent only cares about the most immediate reward.

# MDPs and One-Step Dynamics

• The state space 𝒮 is the set of all (nonterminal) states.
• In episodic tasks, we use 𝒮⁺ to refer to the set of all states, including terminal states.
• The action space 𝒜 is the set of possible actions. (Alternatively, 𝒜(s) refers to the set of possible actions
available in state s ∈ 𝒮)
• (Please see Part 2 to review how to specify the reward signal in the recycling robot example.)
• The one-step dynamics of the environment determine how the environment decides the state and reward at every time
step. The dynamics can be defined by specifying p(s', r|s, a) = ℙ(St+1 = s', Rt+1 = r|St = s, At = a) for each possible
s', r, s, and a.
• A (finite) Markov Decision Process (MDP) is defined by:
- a (finite) set of states 𝒮 or 𝒮⁺ (in the case of an episodic task)
- a (finite) set of actions 𝒜
- a set of rewards ℛ
- the one-step dynamics of the environment
- the discount rate γ ∈ [0,1]

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

<br>

![localImage](images/summary.png)

<br>
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"

[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Value-Based Methods

![Trained Agents][image1]

This repository contains material related to Udacity's Value-based Methods course.

## Table of Contents

### Tutorials

The tutorials lead you through implementing various algorithms in reinforcement learning. All of the code is in
PyTorch (v0.4) and Python 3.

* [Deep Q-Network](https://github.com/udacity/Value-based-methods/tree/main/dqn): Explore how to use a Deep Q-Network (
  DQN) to navigate a space vehicle without crashing.

### Labs / Projects

The labs and projects can be found below. All of the projects use rich simulation environments
from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents).

* [Navigation](https://github.com/udacity/Value-based-methods/tree/main/p1_navigation): In the first project, you will
  train an agent to collect yellow bananas while avoiding blue bananas.

### Resources

* [Cheatsheet](https://github.com/udacity/Value-based-methods/tree/main/cheatsheet): You are encouraged to
  use [this PDF file](https://github.com/udacity/Value-based-methods/blob/main/cheatsheet/cheatsheet.pdf) to guide your
  study of reinforcement learning.

## OpenAI Gym Benchmarks

### Box2d

- `LunarLander-v2`
  with [Deep Q-Networks (DQN)](https://github.com/udacity/Value-based-methods/blob/main/dqn/solution/Deep_Q_Network_Solution.ipynb) |
  solved in 1504 episodes

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

2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI
   gym.
    - Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder. Then, install several
   dependencies.

```bash
git clone https://github.com/udacity/Value-based-methods.git
cd Value-based-methods/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd`
   environment.

```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel`
   menu.

![Kernel][image2]

## Want to learn more?

<p align="center">Come learn with us in the <a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">Deep Reinforcement Learning Nanodegree</a> program at Udacity!</p>

<p align="center"><a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">
 <img width="503" height="133" src="https://user-images.githubusercontent.com/10624937/42135812-1829637e-7d16-11e8-9aa1-88056f23f51e.png"></a>
</p>
