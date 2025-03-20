# S-1: Foundations and Classical Methods in Reinforcement Learning

# C-1: Foundations of Reinforcement Learning

1. Reinforcement Learning Fundamentals

    - The RL Framework and Core Components
    - Agent-Environment Interaction
    - Real-World Applications of RL
    - Reward Signals and Design Principles

2. Mathematical Foundations

    - Returns and Discounting
    - State and Action Value Functions
    - Policies and Optimal Behavior
    - The Bellman Equations

3. Markov Decision Processes
    - The Markov Property
    - MDP Formalization
    - One-Step Dynamics
    - Episodic vs Continuing Tasks



#### Reinforcement Learning Fundamentals

##### The RL Framework and Core Components

Reinforcement learning (RL) represents a fundamentally different paradigm of learning compared to supervised or
unsupervised approaches. At its core, RL is concerned with how intelligent agents should take actions in an environment
to maximize cumulative reward over time. This approach draws inspiration from behavioral psychology, particularly the
concept of operant conditioning, where behavior is modified through rewards and punishments.

The essential components of an RL framework include:

<div align="center"> <img src="images/agent.png" width="600" height="auto"> <p style="color: #555;">Figure: Agent-Environment interaction in reinforcement learning</p> </div>

1. **Agent**: The decision-maker that learns from and interacts with the environment. The agent represents the learning
   algorithm that aims to achieve a specific goal.
2. **Environment**: The external system with which the agent interacts. The environment presents situations (states) to
   the agent, receives its actions, and provides feedback in the form of rewards.
3. **State** $(S_t)$: A representation of the current situation or condition of the environment at time $t$. States can
   be fully or partially observable, discrete or continuous.
4. **Action** $(A_t)$: The set of possible decisions the agent can make. The agent's policy determines which action to
   take in a given state.
5. **Reward** $(R_t)$: The feedback signal that indicates the immediate value of the current state-action transition.
   The reward is the fundamental mechanism that guides the learning process.
6. **Policy** $(\pi)$: The agent's strategy for selecting actions in different states. It can be deterministic (always
   choosing the same action in a given state) or stochastic (selecting actions according to a probability distribution).
7. **Value Function**: An estimation of the expected cumulative future reward from a given state or state-action pair.
8. **Model**: An optional component that represents the agent's understanding of the environment dynamics—how states
   transition and rewards are generated.

The RL process follows a cyclical pattern:

1. The agent observes the current state $S_t$ of the environment.
2. Based on this state, the agent selects an action $A_t$ according to its policy.
3. The environment transitions to a new state $S_{t+1}$ and provides a reward $R_{t+1}$.
4. The agent updates its knowledge (policy and/or value functions) based on this experience.
5. The process repeats from step 1 with the new state.

This framework forms the basis for all reinforcement learning algorithms, regardless of their specific implementation
details or applications.

### Agent-Environment Interaction

The interaction between the agent and environment constitutes the foundation of the reinforcement learning paradigm.
This interaction is sequential and continuous, forming what we call the agent-environment loop:

1. At each discrete time step $t$, the agent receives a representation of the environment's state $S_t \in \mathcal{S}$,
   where $\mathcal{S}$ is the set of all possible states.
2. Based on this state, the agent selects an action $A_t \in \mathcal{A}(S_t)$, where $\mathcal{A}(S_t)$ is the set of
   actions available in state $S_t$.
3. As a consequence of this action, the environment transitions to a new state $S_{t+1}$ and generates a reward
   $R_{t+1}$.
4. The agent receives the new state $S_{t+1}$ and reward $R_{t+1}$, which it uses to update its policy or value
   estimates.

This interaction can be represented formally as a sequence:

$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, \ldots$

A crucial aspect of this interaction is the timing of events. The reward $R_{t+1}$ is the consequence of the action
$A_t$ taken in state $S_t$. It is received alongside the next state $S_{t+1}$. This temporal relationship is essential
for understanding the credit assignment problem in RL—determining which actions were responsible for which rewards.

The agent-environment boundary is conceptual rather than physical. For example, in a robot learning to walk, the motors
and mechanical components might be considered part of the agent, while the external physical world constitutes the
environment. However, the boundary could be drawn differently depending on the problem formulation.

A unique characteristic of RL is that the agent must learn from its own experiences—it does not have access to labeled
examples of correct behavior as in supervised learning. Instead, it must explore the environment, observe the
consequences of its actions, and adapt its behavior to maximize reward. This exploration-exploitation tradeoff is a
central challenge in reinforcement learning.

### Real-World Applications of RL

Reinforcement learning has demonstrated remarkable success across diverse domains, showcasing its versatility and
potential. Some prominent applications include:

1. **Game Playing**: RL has achieved superhuman performance in games ranging from classic board games like Chess and Go
   (AlphaGo, AlphaZero) to video games like Atari and StarCraft. These games provide well-defined environments with
   clear rules and objectives, making them ideal testbeds for RL algorithms.
2. **Robotics**: RL enables robots to learn complex behaviors through interaction with their physical environment.
   Applications include:
    - Locomotion (walking, running, jumping)
    - Manipulation (grasping, stacking, assembly)
    - Navigation in complex environments
    - Adaptive control for changing conditions
3. **Autonomous Vehicles**: RL contributes to decision-making systems for self-driving cars, helping them navigate
   complex traffic scenarios, plan routes, and adapt to diverse driving conditions.
4. **Healthcare**:
    - Personalized treatment plans
    - Optimizing medication dosages
    - Clinical trial design
    - Automated diagnosis from medical images
5. **Finance**:
    - Portfolio management
    - Algorithmic trading
    - Risk management
    - Fraud detection
6. **Resource Management**:
    - Energy grid optimization
    - Data center cooling systems
    - Network traffic management
    - Inventory control
7. **Recommendation Systems**:
    - Content recommendation (videos, articles)
    - Product recommendation in e-commerce
    - Dynamic pricing strategies
    - User experience optimization
8. **Natural Language Processing**:
    - Dialogue systems
    - Text summarization
    - Machine translation
    - Question-answering systems

These applications demonstrate that RL is particularly effective in domains with the following characteristics:

- Sequential decision-making problems
- Clear reward signals (or constructible reward functions)
- Environments that are too complex to model explicitly
- Problems where exploration can be safely conducted
- Tasks that benefit from continuous improvement through experience

As technology advances, we can expect reinforcement learning to expand into new domains and tackle increasingly complex
real-world challenges.

### Reward Signals and Design Principles

The reward signal is the primary means by which we specify the goal of a reinforcement learning problem. It defines what
we want the agent to achieve, not how to achieve it. Proper reward design is crucial for successful RL applications.

#### Reward Function Principles

1. **Delayed Reward**: Unlike supervised learning, rewards in RL may be delayed in time from the actions that caused
   them. This creates the temporal credit assignment problem—determining which actions in a sequence led to a given
   reward.
2. **Sparsity vs. Density**: Rewards can be sparse (provided only upon completing a task) or dense (provided throughout
   the learning process). Sparse rewards are more natural but harder to learn from, while dense rewards provide more
   feedback but require careful design.
3. **Reward Shaping**: The practice of providing intermediate rewards to guide learning. This must be done carefully to
   avoid subverting the original goal.

<div align="center"> <img src="images/reward.png" width="600" height="auto"> <p style="color: #555;">Figure: Reward function decomposition for robot locomotion</p> </div>

Consider this reward function for training a robot to walk:

$r = \min(v_x, v_{max}) - 0.005(v_y^2 + v_z^2) - 0.05y^2 - 0.02||u||^2 + 0.02$

This function elegantly combines multiple objectives:

- Encourage forward velocity (up to a maximum)
- Discourage sideways and vertical motion
- Penalize deviation from the center path
- Minimize energy consumption (through torque)
- Provide a small bonus for remaining upright

#### Reward Design Challenges

1. **Reward Hacking**: Agents may find unexpected ways to maximize reward without satisfying the designer's intent. For
   example, if a cleaning robot is rewarded for not seeing dirt, it might learn to close its eyes or avoid rooms
   altogether.
2. **Misspecification**: The reward function may not accurately capture what we truly value. For instance, rewarding a
   chess agent solely for capturing pieces might lead to reckless play rather than winning the game.
3. **Short-Term vs. Long-Term**: Balancing immediate feedback with long-term goals is difficult. Too much focus on
   short-term rewards can prevent discovery of better long-term strategies.
4. **Transferability**: Ideally, reward functions should generalize across different environments and tasks, but
   designing such transferable rewards is challenging.

#### Effective Reward Design Principles

1. **Alignment with True Objectives**: Ensure the reward function truly captures what you want the agent to accomplish.
2. **Simplicity**: When possible, use simple reward functions that are less prone to misspecification and exploitation.
3. **Careful Shaping**: If using reward shaping, ensure shaped rewards don't introduce behaviors that conflict with the
   original goal.
4. **Robustness**: Design rewards that are robust to minor variations in the environment and agent behavior.
5. **Exploration Encouragement**: Consider intrinsic rewards to encourage exploration, especially in sparse-reward
   environments.

Understanding and applying these principles for reward signal design is essential for developing effective reinforcement
learning systems that achieve their intended goals while avoiding unintended consequences.

#### Mathematical Foundations

### Returns and Discounting

In reinforcement learning, the agent's objective is to maximize the expected cumulative reward over time, not just
immediate rewards. This cumulative measure is called the **return**.

#### Types of Returns

1. **Finite-Horizon Return (Undiscounted Sum)**: The sum of rewards up to a final time step T:

    $$G_t = R_{t+1} + R_{t+2} + R_{t+3} + ... + R_T$$

    This is appropriate for episodic tasks with a clear endpoint, such as games with a finite number of moves.

2. **Infinite-Horizon Discounted Return**: For continuing tasks with no natural endpoint, we use a discounted sum:

    $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

    Where $\gamma \in [0,1]$ is the discount factor.

#### The Role of Discounting

Discounting serves several important purposes in reinforcement learning:

1. **Mathematical Convergence**: With $\gamma < 1$, the infinite sum converges to a finite value even for continuing
   tasks, provided the reward sequence is bounded.
2. **Uncertainty Handling**: Future rewards are less certain than immediate ones. Discounting reflects this uncertainty
   by giving less weight to rewards further in the future.
3. **Present Value Modeling**: Aligns with economic principles where immediate rewards are more valuable than delayed
   ones of the same magnitude.
4. **Computational Tractability**: Makes the learning problem more manageable by limiting the effective planning
   horizon.

#### Effects of Different Discount Factors

The choice of discount factor significantly impacts agent behavior:

- **$\gamma = 0$**: Myopic agent, considers only immediate rewards
- **$\gamma$ close to 0**: Strongly present-focused, may miss valuable long-term strategies
- **$\gamma$ close to 1**: Far-sighted agent, values future rewards almost as much as immediate ones
- **$\gamma = 1$**: Completely far-sighted, appropriate only for finite-horizon problems

#### Recursive Relationship

A key property of the return is its recursive relationship:

$$G_t = R_{t+1} + \gamma G_{t+1}$$

This recursive formulation is fundamental to dynamic programming approaches in RL and underlies the Bellman equations
we'll examine later.

#### Example: Robot Navigation

Consider a robot navigating a maze:

- Immediate reward for each step: -1 (cost of time)
- Reward for reaching the goal: +100
- Discount factor: $\gamma = 0.9$

With discounting, the value of reaching the goal decreases with distance:

- 1 step away: $0.9 \times 100 = 90$ minus step cost
- 10 steps away: $0.9^{10} \times 100 \approx 35$ minus step costs

This naturally encourages finding shorter paths to the goal.

Understanding returns and discounting provides the mathematical foundation for defining what we want RL agents to
optimize, which is essential for developing effective learning algorithms.

### State and Action Value Functions

Value functions are central to reinforcement learning as they estimate how good it is for an agent to be in a given
state or to take a specific action in a given state. These functions provide a way to evaluate the expected future
rewards.

#### State-Value Function

The state-value function $V_\pi(s)$ represents the expected return when starting in state $s$ and following policy $π$
thereafter:

$$V_\pi(s) = \mathbb{E}*\pi[G_t | S_t = s] = \mathbb{E}*\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s\right]$$

This function answers the question: "How good is it to be in state $s$ when following policy $π$?"

<div align="center"> <img src="images/policy.png" width="600" height="auto"> <p style="color: #555;">Figure: Comparison of state-value and action-value representations</p> </div>

#### Action-Value Function

The action-value function $Q_\pi(s,a)$ represents the expected return when starting in state $s$, taking action $a$, and
thereafter following policy $π$:

$$Q_\pi(s,a) = \mathbb{E}*\pi[G_t | S_t = s, A_t = a] = \mathbb{E}*\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s, A_t = a\right]$$

This function answers the question: "How good is it to take action $a$ in state $s$ when following policy $π$
afterward?"

#### Relationship Between Value Functions

There's a fundamental relationship between state-value and action-value functions:

$$V_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q_\pi(s,a)$$

This equation shows that the value of a state is the expected value of the actions that might be taken in that state,
weighted by their probabilities under policy $π$.

#### Value Functions as Tables vs. Function Approximation

For small, discrete state spaces, value functions can be represented as tables with one entry per state or state-action
pair. For large or continuous spaces, function approximation techniques are necessary.

#### Comparing State-Value and Action-Value Functions

| Aspect      | State-Value Function (V)            | Action-Value Function (Q)         |
| ----------- | ----------------------------------- | --------------------------------- |
| Input       | State only                          | State-action pair                 |
| Output      | Expected return from state          | Expected return from state-action |
| Usage       | Better for policy evaluation        | Better for action selection       |
| Information | Requires model for action selection | Can select actions directly       |
| Application | Model-based methods                 | Model-free methods                |
| Memory      | Stores \|S\| values                 | Stores \|S\|×\|A\| values         |

#### Optimal Value Functions

The optimal state-value function $V_*(s)$ gives the maximum value achievable by any policy for each state:

$$V_*(s) = \max_\pi V_\pi(s)$$

Similarly, the optimal action-value function $Q_*(s,a)$ gives the maximum expected return for each state-action pair:

$$Q_*(s,a) = \max_\pi Q_\pi(s,a)$$

These optimal value functions are central to finding optimal policies in reinforcement learning. If we know $Q_*$, the
optimal policy can be derived by selecting the action with the highest Q-value in each state.

Value functions provide a way to evaluate states and actions, guiding the agent toward better policies. Their estimation
is a primary focus of many reinforcement learning algorithms.

### Policies and Optimal Behavior

A policy is the agent's strategy for choosing actions in each state. It defines the agent's behavior and is central to
the reinforcement learning process.

#### Types of Policies

1. **Deterministic Policy**: Maps each state to a single action $$\pi(s) = a$$

    This means that in state $s$, the agent always selects action $a$.

2. **Stochastic Policy**: Maps each state to a probability distribution over actions $$\pi(a|s) = P(A_t = a | S_t = s)$$

    This means that in state $s$, the agent selects action $a$ with probability $\pi(a|s)$.

#### Policy Evaluation and Improvement

The process of learning in RL often involves alternating between two steps:

1. **Policy Evaluation**: Computing the value function for a given policy
2. **Policy Improvement**: Updating the policy to be greedy with respect to the current value function

This iterative process is the foundation of many RL algorithms and is guaranteed to converge to the optimal policy under
appropriate conditions.

#### Optimal Policy

An optimal policy, denoted $\pi_*$, maximizes the expected return from all states:

$$\pi_* = \arg\max_\pi V_\pi(s), \forall s \in \mathcal{S}$$

Multiple optimal policies may exist, but they all share the same optimal value functions $V_*$ and $Q_*$.

<div align="center"> <img src="images/optimal_policy.png" width="600" height="auto"> <p style="color: #555;">Figure: Optimal policy and value function in the game of Blackjack</p> </div>

#### Deriving Optimal Policies

If we know the optimal action-value function $Q_*$, we can directly derive an optimal deterministic policy:

$$\pi_*(s) = \arg\max_a Q_*(s, a)$$

This means selecting the action with the highest Q-value in each state.

#### Exploration-Exploitation Tradeoff

A key challenge in RL is balancing:

- **Exploitation**: Using current knowledge to maximize immediate reward
- **Exploration**: Gathering more information that may lead to better rewards in the future

Common policy types addressing this tradeoff include:

1. **ε-greedy Policy**: With probability $1-\varepsilon$, select the greedy action; with probability $\varepsilon$,
   select a random action.

    $$\pi(a|s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|}, & \text{if } a = \arg\max_{a'} Q(s,a') \ \frac{\varepsilon}{|\mathcal{A}(s)|}, & \text{otherwise} \end{cases}$$

2. **Softmax Policy**: Actions are selected with probabilities proportional to their estimated values.

    $$\pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}$$

    Where $\tau$ is a temperature parameter controlling exploration.

#### Policy Gradient Methods

Instead of learning value functions and deriving policies from them, policy gradient methods directly parametrize and
optimize the policy:

$$\pi_\theta(a|s) = P(A_t = a | S_t = s, \theta)$$

Where $\theta$ is a parameter vector that is adjusted to maximize expected return:

$$J(\theta) = \mathbb{E}*{\pi*\theta}[G_0]$$

These methods are particularly useful for continuous action spaces or when we want to learn stochastic policies.

#### Constrained Policies

In practical applications, we often need to add constraints to policies for safety, resource limits, or other
considerations. Constrained RL formulates this as:

$$\max_\pi \mathbb{E}*\pi[G_0] \text{ subject to } \mathbb{E}*\pi[C_i] \leq d_i, i = 1,2,...,m$$

Where $C_i$ are constraint functions and $d_i$ are threshold values.

Policies form the core of an agent's decision-making process in reinforcement learning. Understanding their properties
and how to optimize them is essential for developing effective RL solutions.

### The Bellman Equations

The Bellman equations are fundamental recursive relationships that characterize value functions in reinforcement
learning. Named after Richard Bellman, these equations express the relationship between the value of a state (or
state-action pair) and the values of its successor states.

#### Bellman Expectation Equation for State Values

For a given policy $\pi$, the Bellman expectation equation for the state-value function is:

$$V_\pi(s) = \sum_{a} \pi(a|s) \sum_{s',r} p(s',r|s,a) [r + \gamma V_\pi(s')]$$

This equation can be broken down as follows:

- $\pi(a|s)$ is the probability of taking action $a$ in state $s$ under policy $\pi$
- $p(s',r|s,a)$ is the probability of transitioning to state $s'$ and receiving reward $r$ after taking action $a$ in
  state $s$
- $r + \gamma V_\pi(s')$ is the immediate reward plus the discounted value of the next state

Intuitively, this equation says: "The value of a state is the expected immediate reward plus the expected discounted
value of the next state, given that we follow policy $\pi$."

#### Bellman Expectation Equation for Action Values

Similarly, for action values:

$$Q_\pi(s,a) = \sum_{s',r} p(s',r|s,a) [r + \gamma \sum_{a'} \pi(a'|s') Q_\pi(s',a')]$$

This says: "The value of taking action $a$ in state $s$ is the expected immediate reward plus the expected discounted
value of taking actions according to policy $\pi$ in the next state."

#### Bellman Optimality Equation for State Values

The Bellman optimality equation for the optimal state-value function is:

$$V_*(s) = \max_a \sum_{s',r} p(s',r|s,a) [r + \gamma V_*(s')]$$

This equation states that the optimal value of a state is the expected return of taking the best action in that state.

#### Bellman Optimality Equation for Action Values

For the optimal action-value function:

$$Q_*(s,a) = \sum_{s',r} p(s',r|s,a) [r + \gamma \max_{a'} Q_*(s',a')]$$

This equation states that the optimal value of a state-action pair is the expected immediate reward plus the expected
discounted value of the next state's best action.

#### Key Properties of Bellman Equations

1. **Recursive Nature**: Value functions are defined in terms of themselves, creating a system of equations.
2. **Fixed Point Solution**: The true value functions are the fixed points of these equations. Solving for these fixed
   points is a key part of many RL algorithms.
3. **Bootstrapping**: Values are estimated using other value estimates, a concept called bootstrapping.
4. **Connection to Dynamic Programming**: Bellman equations provide the theoretical foundation for dynamic programming
   solutions to RL problems.

#### Solving Bellman Equations

There are several approaches to solving Bellman equations:

1. **Analytical Solution**: Possible only for very small, simple MDPs.
2. **Iterative Solution**: Methods like value iteration repeatedly apply the Bellman operator to converge to the
   solution.
3. **Sampling-Based Approximation**: Monte Carlo and temporal-difference learning use samples to approximate solutions.
4. **Function Approximation**: For large or continuous state spaces, function approximators like neural networks are
   used to represent value functions.

The Bellman equations are not just theoretical constructs; they form the basis for practical algorithms in reinforcement
learning. Understanding these equations provides insight into how RL algorithms work and why they converge to optimal
solutions.

## 3. Markov Decision Processes

### The Markov Property

The Markov property is a fundamental concept in reinforcement learning that enables mathematical tractability and
efficient algorithm design. It describes a specific constraint on how the future depends on the past.

#### Definition of the Markov Property

A state $S_t$ is Markovian if and only if:

$$P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0) = P(S_{t+1} | S_t, A_t)$$

In other words, the future state $S_{t+1}$ depends only on the current state $S_t$ and action $A_t$, not on the history
of states and actions that preceded them. The current state encapsulates all relevant information from the history.

#### Implications of the Markov Property

1. **State Sufficiency**: The current state provides sufficient statistics for the future. Once we know the current
   state, the history becomes irrelevant for predicting the future.
2. **Memory-less System**: The system has no "memory" beyond what is captured in the state. This simplifies both the
   mathematical analysis and algorithm design.
3. **One-Step Dynamics**: We need only model transitions from one state to the next, rather than considering sequences
   of states.
4. **Compactness**: Markovian states allow for compact representations of the environment dynamics.

#### When the Markov Property Doesn't Hold

In many practical applications, the observed state may not satisfy the Markov property. For example:

1. **Partial Observability**: The agent doesn't have access to the complete state information.
2. **Incomplete State Representation**: The chosen state representation omits relevant information.
3. **Delayed Effects**: Actions have consequences that only become apparent after multiple time steps.

#### Dealing with Non-Markovian Environments

When faced with non-Markovian environments, several approaches are possible:

1. **State Augmentation**: Include additional information in the state to make it Markovian (e.g., including velocity
   along with position).
2. **History-Based Approaches**: Incorporate history into the state representation (e.g., using the last $n$
   observations).
3. **Recurrent Neural Networks**: Use architectures with memory capabilities to implicitly learn temporal dependencies.
4. **Partially Observable MDPs (POMDPs)**: Formally model the uncertainty in state observations.

#### The Markov Property and Value Functions

Value functions in reinforcement learning rely on the Markov property. For a state to have a well-defined value,
independent of history, it must be Markovian. This is why proper state representation is crucial for effective
reinforcement learning.

The Markov property provides the mathematical foundation for modeling sequential decision processes as Markov Decision
Processes (MDPs), which we'll explore in the next section. Understanding this property is essential for designing
effective state representations and choosing appropriate RL algorithms for a given problem.

### MDP Formalization

A Markov Decision Process (MDP) provides a formal framework for modeling sequential decision-making problems where an
agent interacts with an environment. MDPs are fundamental to reinforcement learning as they define the problem structure
to which RL algorithms are applied.

#### Formal Definition of an MDP

An MDP is defined as a tuple $(S, A, P, R, \gamma)$ where:

1. **State Space $S$**: The set of all possible states. This can be:

    - Finite or infinite
    - Discrete or continuous
    - Fully or partially observable

2. **Action Space $A$**: The set of all possible actions available to the agent. Like the state space, this can be
   finite/infinite and discrete/continuous. Sometimes denoted as $A(s)$ to indicate state-dependent action sets.

3. **Transition Function $P$**: Defines the dynamics of the environment: $$P(s'|s,a) = Pr(S_{t+1}=s'|S_t=s, A_t=a)$$

    This gives the probability of transitioning to state $s'$ after taking action $a$ in state $s$.

4. **Reward Function $R$**: Defines the immediate reward received after transitions:
   $$R(s,a,s') = \mathbb{E}[R_{t+1}|S_t=s, A_t=a, S_{t+1}=s']$$

    Sometimes simplified as $R(s,a)$ when the reward depends only on the current state and action.

5. **Discount Factor $\gamma$**: A value in $[0,1]$ that determines the present value of future rewards. It balances
   immediate versus future rewards.

#### Types of MDPs

1. **Finite MDPs**: Have finite state and action spaces. Most theoretical guarantees in RL apply to finite MDPs.
2. **Infinite MDPs**: Have infinite state or action spaces (e.g., continuous spaces). Require function approximation
   techniques.
3. **Episodic MDPs**: Tasks that end in terminal states. Episodes have a definite conclusion.
4. **Continuing MDPs**: Tasks that continue indefinitely without terminal states. Require discounting for well-defined
   returns.

#### Optimal Solutions to MDPs

The goal in an MDP is to find an optimal policy $\pi^*$ that maximizes expected cumulative reward. This policy
satisfies:

$$\pi^* = \arg\max_\pi V^\pi(s) \text{ for all } s \in S$$

Finding this optimal policy involves computing the optimal value functions:

1. **Optimal State-Value Function**: $$V^*(s) = \max_\pi V^\pi(s) \text{ for all } s \in S$$
2. **Optimal Action-Value Function**: $$Q^*(s,a) = \max_\pi Q^\pi(s,a) \text{ for all } s \in S, a \in A$$

Once we know $Q^*$, the optimal deterministic policy can be derived as: $$\pi^*(s) = \arg\max_a Q^*(s,a)$$

#### MDP Solving Methods

MDPs can be solved using various approaches:

1. **Dynamic Programming**: When the model (P and R) is known, methods like value iteration and policy iteration can
   find the optimal policy.
2. **Model-Free Methods**: When the model is unknown, methods like Q-learning or SARSA learn from experience without
   requiring a model.
3. **Model-Based RL**: Learns a model of the environment from experience, then uses planning algorithms to find an
   optimal policy.

The MDP formalization provides a rigorous mathematical foundation for reinforcement learning. It allows us to precisely
define the problem, analyze solution properties, and develop algorithms with theoretical guarantees. Understanding MDPs
is essential for mastering the fundamental principles of reinforcement learning.

### One-Step Dynamics

One-step dynamics define how the environment responds to the agent's actions at each time step. These dynamics are
central to the Markov Decision Process (MDP) framework and describe the immediate consequences of actions in terms of
state transitions and rewards.

#### Probability Distributions of Transitions

The one-step dynamics of an MDP are fully defined by the probability distribution:

$$p(s', r | s, a) = Pr(S_{t+1}=s', R_{t+1}=r | S_t=s, A_t=a)$$

This joint distribution specifies the probability of transitioning to state $s'$ and receiving reward $r$ after taking
action $a$ in state $s$.

<div align="center"> <img src="images/robot.png" width="600" height="auto"> <p style="color: #555;">Figure: One-step dynamics for robot battery states with different actions</p> </div>

From this fundamental distribution, we can derive several useful quantities:

1. **State-Transition Probabilities**: $$p(s' | s, a) = Pr(S_{t+1}=s' | S_t=s, A_t=a) = \sum_r p(s', r | s, a)$$

    This gives the probability of the next state, regardless of reward.

2. **Expected Rewards**: $$r(s, a) = \mathbb{E}[R_{t+1} | S_t=s, A_t=a] = \sum_{s', r} r \cdot p(s', r | s, a)$$

    This gives the expected immediate reward for a state-action pair.

3. **Expected Rewards for State-Action-Next-State**:
   $$r(s, a, s') = \mathbb{E}[R_{t+1} | S_t=s, A_t=a, S_{t+1}=s'] = \sum_r r \cdot \frac{p(s', r | s, a)}{p(s' | s, a)}$$

    This gives the expected reward when a specific transition occurs.

#### The Markov Property in One-Step Dynamics

The one-step dynamics embody the Markov property by depending only on the current state and action, not on the history:

$$(S_{t+1}, R_{t+1}) \$$(S_{t+1}, R_{t+1})$$ depend only on $$(S_t, A_t)$$ and not on any earlier states or actions.
This property allows us to define the environment's dynamics concisely and enables the mathematical tractability of
MDPs.

#### Example: Robot Battery Management

Consider a robot with two battery states: "high" and "low". The robot has three possible actions: "wait", "search", and
"recharge". The one-step dynamics could be represented as:

<div align="center"> <img src="images/robot.png" width="600" height="auto"> <p style="color: #555;">Figure: State transition probabilities for robot battery management</p> </div>

For instance, when the robot is in the "high" battery state and takes the "search" action:

- With probability 0.7, it remains in "high" state and receives a reward of 4
- With probability 0.3, it transitions to "low" state and receives a reward of 4

These dynamics completely specify how the environment responds to each action in each state, which is essential for
planning and learning optimal policies.

#### Tabular Representation

For finite MDPs, one-step dynamics can be represented in tabular form, with entries for each (s, a, s', r) combination.
This representation is complete but can become unwieldy for large state and action spaces.

#### Factored and Structured Representations

For large or complex environments, the one-step dynamics can often be represented more compactly using structured
representations:

1. **Factored MDPs**: The state is represented as a vector of variables, and transitions for each variable depend only
   on a subset of other variables.
2. **Dynamic Bayesian Networks**: Graphical models that capture conditional dependencies in state transitions.
3. **Parametric Models**: Mathematical functions with parameters that determine transition probabilities (e.g., physical
   models in robotics).

#### Stochastic vs. Deterministic Dynamics

One-step dynamics can be:

1. **Deterministic**: Each state-action pair leads to exactly one next state and reward.
   $$s_{t+1} = f(s_t, a_t) \text{ and } r_{t+1} = R(s_t, a_t)$$
2. **Stochastic**: State transitions and rewards have probability distributions. $$P(s_{t+1}, r_{t+1} | s_t, a_t)$$

Most real-world problems involve stochastic dynamics due to inherent randomness or incomplete state information.

Understanding one-step dynamics is crucial for developing and analyzing reinforcement learning algorithms. They form the
foundation for both model-based approaches (which directly use these dynamics) and model-free approaches (which
implicitly learn these dynamics through experience).

### Episodic vs Continuing Tasks

Reinforcement learning problems can be categorized into two fundamental types based on their temporal structure:
episodic tasks and continuing tasks. This distinction has important implications for problem formulation, algorithm
design, and evaluation.

#### Episodic Tasks

Episodic tasks are characterized by well-defined starting and ending points, collectively forming what we call an
episode.

**Key Characteristics:**

1. **Terminal States**: Episodes conclude when the agent reaches a terminal state.
2. **Finite Horizon**: There is a natural end to the interaction sequence.
3. **Episode Reset**: After reaching a terminal state, the environment is reset to an initial state for a new episode.
4. **Cumulative Return**: The return is calculated over the finite number of steps in the episode:
   $$G_t = R_{t+1} + R_{t+2} + ... + R_T$$ where T is the terminal time step.

**Examples of Episodic Tasks:**

- Games with clear endings (chess, Go, video game levels)
- Navigation tasks with goal locations
- Assembly tasks with completion criteria
- Trading simulations with fixed time horizons

In episodic tasks, learning progress is often measured by evaluating performance over multiple episodes, such as the
average return per episode or the success rate in completing the task.

#### Continuing Tasks

Continuing tasks have no natural endpoint and theoretically continue indefinitely.

**Key Characteristics:**

1. **No Terminal States**: The agent-environment interaction continues without end.
2. **Infinite Horizon**: The sequence of states, actions, and rewards is unbounded.
3. **Discounted Return**: To ensure the return is finite, we use discounting:
   $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$ where
   $\gamma \in [0,1)$ is the discount factor.
4. **Average Reward Formulation**: An alternative to discounting is to optimize the long-term average reward:
   $$\overline{R} = \lim_{n\to\infty} \frac{1}{n} \sum_{t=1}^{n} \mathbb{E}[R_t]$$

**Examples of Continuing Tasks:**

- Process control in manufacturing
- Power grid management
- Stock portfolio management
- Autonomous driving in operational settings
- Ongoing recommendation systems

<div align="center"> <img src="images/summary.png" width="600" height="auto"> <p style="color: #555;">Figure: Summary comparison of episodic vs continuing tasks</p> </div>

#### Practical Considerations

1. **Hybrid Approaches**: Some problems can be formulated as either episodic or continuing. For example, a robot
   navigation task could be treated as episodic with explicit goals or continuing with ongoing operation.
2. **Artificially Episodic**: Continuing tasks are sometimes approximated as episodic by imposing artificial episode
   boundaries (e.g., dividing a continuous process into fixed-length segments).
3. **Discount Factor Selection**: In episodic tasks with discounting, $\gamma$ can be set to 1 if episodes are
   guaranteed to terminate. In continuing tasks, $\gamma < 1$ is necessary for mathematical convergence.
4. **Algorithm Applicability**: Some algorithms are specifically designed for either episodic or continuing tasks, while
   others can handle both with appropriate modifications.
5. **Evaluation Metrics**: Performance metrics differ between episodic tasks (e.g., success rate, episode length) and
   continuing tasks (e.g., average reward per time step).

Understanding whether a problem is episodic or continuing is a fundamental step in formulating it within the
reinforcement learning framework. This categorization guides choices about return calculation, algorithm selection, and
evaluation procedures.
