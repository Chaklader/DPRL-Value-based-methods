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

# The RL Framework Solution

### State-Value Functions & Bellman Equations

## 1. State-Value Function Under a Policy

### Definition

The state-value function $v_π(s)$ for a policy π is the expected return when starting in state s and following policy π
thereafter:

$v_π(s) \doteq E_π[G_t|S_t=s]$
$= E_π[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}|S_t=s]$

Where:

- $G_t$ is the return
- γ is the discount factor
- π is the policy being evaluated

## 2. Bellman Equation for $v_π$

### Basic Form

$v_π(s) = \sum_a π(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma v_π(s')]$

### Components Breakdown:

1. $π(a|s)$: Probability of taking action a in state s
2. $p(s',r|s,a)$: Probability of transition to s' with reward r
3. $r$: Immediate reward
4. $\gamma v_π(s')$: Discounted value of next state

## 3. Bellman Optimality Equation

### For Optimal State-Value Function

$v_*(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma v_*(s')]$

### For Optimal Action-Value Function

$q_*(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \max_{a'} q_*(s',a')]$

## 4. Key Properties

### Recursive Nature

- Current value depends on future values
- Forms a system of equations
- Solution gives optimal values

### Policy Improvement

Better policy π' can be found by:
$π'(s) = \arg\max_a \sum_{s',r} p(s',r|s,a)[r + \gamma v_π(s')]$

## 5. Practical Applications

### Value Iteration

1. Initialize $v(s)$ arbitrarily
2. Update:
   $v(s) \leftarrow \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma v(s')]$

### Policy Iteration

1. Policy Evaluation: Compute $v_π$
2. Policy Improvement: Update π
3. Repeat until convergence

## 6. Example: Grid World

```textmate
Initial State:
[ 0  0  0 ]
[ 0  x  0 ]
[ 0  0  G ]

Where:
- G: Goal state
- x: Obstacle
- 0: Empty cells
```

Value function might look like:

```textmate
[ 0.5  0.7  0.9 ]
[ 0.3   x   1.0 ]
[ 0.1  0.8  G   ]
```

## 7. Relationship to Dynamic Programming

### Steps:

1. Break down problem into subproblems
2. Store solutions to subproblems
3. Use stored solutions to solve larger problems

## 8. Important Considerations

### Computational Challenges:

1. Large state spaces
2. Continuous states/actions
3. Unknown transition probabilities
4. Exploration vs exploitation

### Practical Solutions:

1. Function approximation
2. Sample-based methods
3. Model-free learning
4. Temporal difference learning

# Bellman Equations in Grid World and MDPs

In this gridworld example, once the agent selects an action,
• it always moves in the chosen direction (contrasting general MDPs where the agent doesn't always have complete control
over what the next state will be), and
• the reward can be predicted with complete certainty (contrasting general MDPs where the reward is a random draw from a
probability distribution).

In this simple example, we saw that the value of any state can be calculated as the sum of the immediate reward and
the (discounted) value of the next state.

Alexis mentioned that for a general MDP, we have to instead work in terms of an expectation, since it's not often the
case that the immediate reward and next state can be predicted with certainty. Indeed, we saw in an earlier lesson that
the reward and next state are chosen according to the one-step dynamics of the MDP. In this case, where the reward r and
next state s' are drawn from a (conditional) probability distribution p(s',r|s,a), the Bellman Expectation Equation (for
vπ) expresses the value of any state s in terms of the expected immediate reward and the expected value of the next
state:

$v_π(s) = E_π[R_{t+1} + \gamma v_π(S_{t+1})|S_t = s]$

# Calculating the Expectation

In the event that the agent's policy π is deterministic, the agent selects action π(s) when in state s, and the Bellman
Expectation Equation can be rewritten as the sum over two variables (s' and r):

$v_π(s) = \sum_{s'∈S^+,r∈R} p(s',r|s,π(s))(r + \gamma v_π(s'))$

In this case, we multiply the sum of the reward and discounted value of the next state $(r + \gamma v_π(s'))$ by its
corresponding probability $p(s',r|s,π(s))$ and sum over all possibilities to yield the expected value.

If the agent's policy π is stochastic, the agent selects action a with probability π(a|s) when in state s, and the
Bellman Expectation Equation can be rewritten as the sum over three variables (s', r, and a):

$v_π(s) = \sum_{s'∈S^+,r∈R,a∈A(s)} π(a|s)p(s',r|s,a)(r + \gamma v_π(s'))$

In this case, we multiply the sum of the reward and discounted value of the next state $(r + \gamma v_π(s'))$ by its
corresponding probability $π(a|s)p(s',r|s,a)$ and sum over all possibilities to yield the expected value.

# There are 3 more Bellman Equations!

In this video, you learned about one Bellman equation, but there are 3 more, for a total of 4 Bellman equations.

All of the Bellman equations attest to the fact that value functions satisfy recursive relationships.

For instance, the Bellman Expectation Equation (for vπ) shows that it is possible to relate the value of a state to the
values of all of its possible successor states.

After finishing this lesson, you are encouraged to read about the remaining three Bellman equations in sections 3.5 and
3.6 of the textbook. The Bellman equations are incredibly useful to the theory of MDPs.

Let me break this down systematically.

# Background

This is a Markov Decision Process (MDP) problem with 9 states (S₁ to S₉), where S₉ is a terminal state. The problem
features:

1. **State Space**: S⁺ = {s₁, s₂, ..., s₉}
2. **Deterministic Policy (π)** given as:
    - π(s₁) = right
    - π(s₂) = right
    - π(s₃) = down
    - π(s₄) = up
    - π(s₅) = right
    - π(s₆) = down
    - π(s₇) = right
    - π(s₈) = right

3. **Rewards**: Shown on transitions in the diagram
    - Most transitions have R = -1 or R = -3
    - Transitions to S₉ have R = 5
    - v_π(s₉) = 0 (terminal state)

4. **Discount Factor**: γ = 1

# Questions and Solutions

## Question 1: What is v_π(s₄)?

**Answer**: 1

**Explanation**:

- From s₄, the policy dictates moving up
- Following the policy: s₄ → s₁ → s₂ → s₃ → s₆ → s₉
- Calculating value:
    - R = -1 (s₄ to s₁)
    - R = -1 (s₁ to s₂)
    - R = -1 (s₂ to s₃)
    - R = -1 (s₃ to s₆)
    - R = 5 (s₆ to s₉)
- Total: -1 + -1 + -1 + -1 + 5 = 1

## Question 2: What is v_π(s₁)?

**Answer**: 2

**Explanation**:

- From s₁, following policy: s₁ → s₂ → s₃ → s₆ → s₉
- Calculating value:
    - R = -1 (s₁ to s₂)
    - R = -1 (s₂ to s₃)
    - R = -1 (s₃ to s₆)
    - R = 5 (s₆ to s₉)
- Total: -1 + -1 + -1 + 5 = 2

Ah, let me help solve Question 3 more systematically using the Bellman Equation.

# Question 3: Which statements are true?

Let's check each statement using the Bellman Equation:

1. v_π(s₆) = -1 + v_π(s₅)
    - Following policy: s₆ → s₉ with R = 5
    - v_π(s₆) = 5 + v_π(s₉) = 5 + 0 = 5
    - This equation is false

2. v_π(s₇) = -3 + v_π(s₈)
    - Following policy: s₇ → s₈ with R = -3
    - v_π(s₇) = -3 + v_π(s₈)
    - v_π(s₈) = -3 + v_π(s₉) = -3
    - Therefore v_π(s₇) = -3 + (-3) = -6
    - This equation is TRUE!

3. v_π(s₁) = -1 + v_π(s₂)
    - Following policy: s₁ → s₂ with R = -1
    - From earlier calculation, v_π(s₁) = 2
    - This equation is TRUE!
    - Because v_π(s₂) = 3 (you get -1, -1, 5 from s₂ → s₃ → s₆ → s₉)
    - So -1 + v_π(s₂) = -1 + 3 = 2 = v_π(s₁)

4. v_π(s₄) = -3 + v_π(s₇)
    - This is not true because policy from s₄ is "up" not down

5. v_π(s₈) = -3 + v_π(s₉)
   ❌ FALSE because:

While s₈ does transition to s₉, the reward is 5, not -3
The equation doesn't match the actual transition dynamics

**Correct Answer**: Statements (2), (3), and (5) are true.

The key is to verify each equation using:

1. The policy-dictated transitions
2. The rewards shown in the diagram
3. The Bellman equation: v_π(s) = R + v_π(s') where s' is the next state following the policy
4. The known values we calculated earlier

Checking with the Bellman equation shows that statements 2, 3, and 5 are consistent with the state-value function and
the transition dynamics of the MDP.

# Optimality in Reinforcement Learning

## 1. Basic Understanding

Optimality in RL refers to achieving the best possible behavior (policy) that maximizes the expected cumulative reward.

### Key Components:

1. **Optimal Value Function (V\*)**
    - Maximum value achievable for each state
    - $V*(s) = \max_\pi V^\pi(s)$ for all s ∈ S

2. **Optimal Action-Value Function (Q\*)**
    - Maximum value achievable for each state-action pair
    - $Q*(s,a) = \max_\pi Q^\pi(s,a)$ for all s ∈ S, a ∈ A

3. **Optimal Policy (π\*)**
    - Policy that achieves the optimal value
    - $\pi*(s) = \arg\max_a Q*(s,a)$

## 2. Bellman Optimality Equations

# State-Value and Action-Value Functions

## State-Value Function (Think of it as "Location Rating")

### What is it?

Imagine you're playing a game of chess. The state-value function is like a score that tells you "how good is my current
position on the board?"

### Real-World Example

- Think of house prices:
    - A house in a good neighborhood (state) has high value
    - The value represents how good it is to be in that location
    - It considers all possible future outcomes from that position

### Key Points

- Only looks at where you are
- Considers long-term benefits
- Based on your overall strategy (policy)

## Action-Value Function (Think of it as "Move Rating")

### What is it?

Using the chess example again, the action-value function tells you "how good is it to make this specific move from my
current position?"

### Real-World Example

- Think of choosing routes while driving:
    - At an intersection (state), you have options to turn left, right, or go straight (actions)
    - Each choice has a different value based on traffic, distance, etc.
    - You want to know the value of each possible move

### Key Points

- Looks at both where you are AND what you're thinking of doing
- Helps directly choose actions
- More detailed than state-value

## Simple Comparison

Think of a GPS Navigation System:

| State-Value (Where you are)                | Action-Value (What move to make)                     |
|--------------------------------------------|------------------------------------------------------|
| "You're in downtown"                       | "Turn left at the next intersection"                 |
| "You're on the highway"                    | "Take exit 34 in 2 miles"                            |
| Tells you how good your location is        | Tells you what specific action to take               |
| Like knowing you're in a good neighborhood | Like knowing which house to buy in that neighborhood |
| General assessment                         | Specific recommendation                              |
| Helps understand situation                 | Helps make decisions                                 |
| Like checking your position                | Like planning your next move                         |

## When to Use Each?

### Use State-Value When:

- You want to understand how good a situation is
- You're evaluating your overall position
- You have a specific strategy in mind

### Use Action-Value When:

- You need to make specific decisions
- You want to compare different options
- You're learning what actions work best

Think of it this way:

- State-Value is like knowing your bank balance
- Action-Value is like knowing what you should buy with that money

### State-Value Function

$V*(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V*(s')]$

### Action-Value Function

$Q*(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \max_{a'} Q*(s',a')]$

## 3. Properties of Optimal Policies

1. **Multiple Optimal Policies**
    - Can have multiple policies achieving V*
    - All optimal policies achieve same optimal value

2. **Deterministic Nature**
    - At least one optimal policy is deterministic
    - No need for randomization if you know optimal values

## 4. Methods to Find Optimal Policies

### Dynamic Programming Methods:

1. **Value Iteration**
   ```textmate
   Initialize V(s) arbitrarily
   Repeat:
       For each s:
           V(s) ← max_a Σ p(s'|s,a)[r + γV(s')]
   ```

2. **Policy Iteration**
   ```textmate
   Initialize π arbitrarily
   Repeat:
       Policy Evaluation: compute V_π
       Policy Improvement: π'(s) ← arg max_a Q(s,a)
   ```

### Model-Free Methods:

1. Q-Learning
2. SARSA
3. Actor-Critic

## 5. Challenges in Finding Optimality

1. **Curse of Dimensionality**
    - State space too large
    - Action space too large

2. **Exploration vs Exploitation**
    - Need to explore to find optimal policy
    - Need to exploit known good actions

3. **Function Approximation**
    - Cannot store all values exactly
    - Need to generalize

## 6. Practical Considerations

### Approximations:

1. Near-optimal policies
2. Local optima
3. Satisficing solutions

### Trade-offs:

1. Computation time vs optimality
2. Memory usage vs accuracy
3. Exploration vs exploitation

## 7. Example

Consider a simple grid world:

```textmate
[ S ][ ][ ]
[ ][ ][ ]
[ ][ ][ G ]
```

Optimal policy might look like:

```textmate
[→][→][↓]
[↓][↘][↓]
[→][→][G]
```

# Action-Value Functions (Q-Functions) in Reinforcement Learning

## 1. Basic Definition

The action-value function (Q-function) measures the expected return starting from state s, taking action a, and then
following policy π:

$Q_π(s,a) = E_π[G_t|S_t=s, A_t=a]$
$= E_π[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}|S_t=s, A_t=a]$

## 2. Key Properties

### Relationship with State-Value Function:

$V_π(s) = \sum_a π(a|s)Q_π(s,a)$

### Optimal Q-Function:

$Q*(s,a) = \max_π Q_π(s,a)$

## 3. Bellman Equations for Q-Functions

### Bellman Expectation Equation:

$Q_π(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'} π(a'|s')Q_π(s',a')]$

### Bellman Optimality Equation:

$Q*(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \max_{a'} Q*(s',a')]$

## 4. Advantages of Q-Functions

1. **Direct Action Selection**
    - Can choose actions without knowing model
    - $a* = \arg\max_a Q*(s,a)$

2. **Model-Free Learning**
    - Don't need transition probabilities
    - Learn directly from experience

3. **Policy Derivation**
    - Optimal policy derived directly:
    - $π*(s) = \arg\max_a Q*(s,a)$

## 5. Q-Learning Example

Consider a simple grid world with Q-values:

```textmate
State A: 
Q(A, right) = 1.0
Q(A, down) = 0.5

State B:
Q(B, right) = 0.8
Q(B, down) = 1.2
```

Best action in:

- State A: go right
- State B: go down

## 6. Practical Applications

### 1. Q-Table

For discrete state-action spaces:

```textmate
Q = {
    (state1, action1): value1,
    (state1, action2): value2,
    ...
}
```

### 2. Deep Q-Networks

For continuous/large spaces:

```textmate
Q(s,a) = NeuralNetwork(state, action)
```

## 7. Important Considerations

### 1. Initialization

- Can start with arbitrary values
- Optimistic initialization encourages exploration

### 2. Updates

Basic Q-learning update:
$Q(s,a) ← Q(s,a) + α[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

### 3. Exploration Strategies

- ε-greedy: With probability ε, choose random action
- Softmax: Choose actions based on relative Q-values
- UCB: Consider uncertainty in Q-value estimates

## 8. Relationship to Other Concepts

### Value Functions:

$V_π(s) = \max_a Q_π(s,a)$ for optimal policy

### Policy Improvement:

New policy: $π'(s) = \arg\max_a Q_π(s,a)$

### Advantage Function:

$A_π(s,a) = Q_π(s,a) - V_π(s)$

# State-Value and Action-Value Functions in RL

## Detailed Explanation

### State-Value Function (vπ)

- Denoted as vπ(s)
- Represents the expected return starting from state s
- Formula: vπ(s) = Eπ[Gt | St = s]
- Evaluates how good it is to be in a state following policy π

### Action-Value Function (qπ)

- Denoted as qπ(s,a)
- Represents the expected return starting from state s, taking action a
- Formula: qπ(s,a) = Eπ[Gt | St = s, At = a]
- Evaluates how good it is to take a specific action in a state following policy π

## Comparison Table

| Aspect                    | State-Value Function (vπ)                         | Action-Value Function (qπ)                                |
|---------------------------|---------------------------------------------------|-----------------------------------------------------------|
| **Input**                 | State only (s)                                    | State and action pair (s,a)                               |
| **Output**                | Value of being in state s                         | Value of taking action a in state s                       |
| **Policy Dependence**     | Values based on following policy π                | Values based on taking action a then following policy π   |
| **Usage**                 | Better for policy evaluation                      | Better for action selection                               |
| **Information Required**  | Needs model or policy for action selection        | Can select actions directly                               |
| **Main Application**      | Model-based methods                               | Model-free methods                                        |
| **Memory Requirements**   | Stores                                            | S                                                         | values | Stores |S|×|A| values |
| **Decision Making**       | Needs additional computation for action selection | Direct action selection possible                          |
| **Bellman Equation Form** | vπ(s) = Σa π(a\|s) qπ(s,a)                        | qπ(s,a) = Σs',r p(s',r\|s,a)[r + γΣa' π(a'\|s')qπ(s',a')] |
| **Optimal Form**          | v*(s) = maxa q*(s,a)                              | q*(s,a) = Σs',r p(s',r\|s,a)[r + γ maxa' q*(s',a')]       |

## Key Relationships

1. State-value can be derived from action-values:
    - vπ(s) = Σa π(a|s)qπ(s,a)

2. Action-value can often lead directly to action selection:
    - a* = argmaxa qπ(s,a)

## Common Use Cases

- **State-Value Function**:
    - Policy evaluation
    - Value iteration
    - Model-based planning

- **Action-Value Function**:
    - Q-learning
    - SARSA
    - Direct policy improvement

<br>

![localImage](images/policy.png)

<br>

### Optimal Policies

# Optimal Action-Value Function and Policy in MDPs

If the state space 𝒮 and action space 𝒜 are finite, we can represent the optimal action-value function q* in a table,
where we have one entry for each possible environment state s ∈ 𝒮 and action a ∈ 𝒜.

The value for a particular state-action pair s, a is the expected return if the agent starts in state s, takes action a,
and then henceforth follows the optimal policy π*.

We have populated some values for a hypothetical Markov decision process (MDP) (where 𝒮 = {s₁, s₂, s₃} and 𝒜 = {a₁, a₂,
a₃}) below.

## First Optimal Action-Value Table (q*):

|    | a₁ | a₂ | a₃ |
|----|----|----|----|
| s₁ | 1  | 2  | -3 |
| s₂ | -2 | 1  | 3  |
| s₃ | 4  | 4  | -5 |

## Same Table with Best Actions Highlighted:

|    | a₁  | a₂  | a₃  |
|----|-----|-----|-----|
| s₁ | 1   | (2) | -3  |
| s₂ | -2  | 1   | (3) |
| s₃ | (4) | (4) | -5  |

## New MDP Question Table:

|    | a₁ | a₂ | a₃ |
|----|----|----|----|
| s₁ | 1  | 3  | 4  |
| s₂ | 2  | 2  | 1  |
| s₃ | 3  | 1  | 1  |

The optimal policy π* must satisfy:

- π*(s₁) = a₂ (or, equivalently, π*(a₂|s₁) = 1)
- π*(s₂) = a₃ (or, equivalently, π*(a₃|s₂) = 1)

For state s₃, with a₁, a₂ ∈ arg max_a∈A(s₃) q*(s₃, a):

- π*(a₁|s₃) = p
- π*(a₂|s₃) = q
- π*(a₃|s₃) = 0

where p, q ≥ 0, and p + q = 1

You learned that once the agent has determined the optimal action-value function q*, it can quickly obtain an optimal
policy π* by setting π*(s) = arg max_a∈A(s) q*(s,a) for all s ∈ 𝒮.

### Quiz

|    | a₁ | a₂ | a₃ |
|----|----|----|----|
| s₁ | 1  | 3  | 4  |
| s₂ | 2  | 2  | 1  |
| s₃ | 3  | 1  | 1  |

Let's analyze each option:

1. "The agent always selects action a₁ in state s₁"

- FALSE: In s₁, a₃ has highest value (4) > a₁ (1)

2. "The agent always selects action a₃ in state s₁"

- TRUE: In s₁, value for a₃ (4) is highest among all actions

3. "The agent is free to select either action a₁ or action a₂ in state s₂"

CORRECT because:
In s₂: both a₁ and a₂ have value 2 (highest)
When multiple actions have equal highest values, the agent can choose either
This is an example of a case where multiple optimal actions exist

4. "The agent must select action a₃ in state s₂"

- FALSE: In s₂, a₃ (1) has lowest value

5. "The agent must select action a₁ in state s₃"

- TRUE: In s₃, a₁ (3) has highest value

6. "The agent is free to select either action a₂ or a₃ in state s₃"

- FALSE: Both have value 1, less than a₁'s value of 3

**Answer**: Options 2 and 5 are correct because:

- For s₁: Must choose a₃ (value 4)
- For s₂: Must choose either a₁ or a₂ (both value 2)
- For s₃: Must choose a₁ (value 3)

Since the question asks for "a potential optimal policy", we need statements that are consistent with the optimal policy
derived from the action-value function using π*(s) = arg max_a∈A(s) q*(s,a).

### Summary

# Policies

- A **deterministic policy** is a mapping π : 𝒮 → 𝒜. For each state s ∈ 𝒮, it yields the action a ∈ 𝒜 that the agent
  will choose while in state s.

- A **stochastic policy** is a mapping π : 𝒮 × 𝒜 → [0,1]. For each state s ∈ 𝒮 and action a ∈ 𝒜, it yields the
  probability π(a|s) that the agent chooses action a while in state s.

# State-Value Functions

- The **state-value function** for a policy π is denoted vπ. For each state s ∈ 𝒮, it yields the expected return if the
  agent starts in state s and then uses the policy to choose its actions for all time steps. That is, vπ(s) =
  𝔼π[Gt|St = s]. We refer to vπ(s) as the value of state s under policy π.

- The notation 𝔼π[·] is borrowed from the suggested textbook, where 𝔼π[·] is defined as the expected value of a random
  variable, given that the agent follows policy π.

# Bellman Equations

- The **Bellman expectation equation** for vπ is:
  vπ(s) = 𝔼π[Rt+1 + γvπ(St+1)|St = s].

# Optimality

- A policy π' is defined to be better than or equal to a policy π if and only if vπ'(s) ≥ vπ(s) for all s ∈ 𝒮.

- An **optimal policy** π* satisfies π* ≥ π for all policies π. An optimal policy is guaranteed to exist but may not be
  unique.

- All optimal policies have the same state-value function v*, called the **optimal state-value function**.

# Action-Value Functions

- The **action-value function** for a policy π is denoted qπ. For each state s ∈ 𝒮 and action a ∈ 𝒜, it yields the
  expected return if the agent starts in state s, takes action a, and then follows the policy for all future time steps.
  That is, qπ(s,a) = 𝔼π[Gt|St = s, At = a]. We refer to qπ(s,a) as the value of taking action a in state s under a
  policy π (or alternatively as the value of the state-action pair s,a).

- All optimal policies have the same action-value function q*, called the **optimal action-value function**.

# Optimal Policies

- Once the agent determines the optimal action-value function q*, it can quickly obtain an optimal policy π* by setting
  π*(s) = arg maxa∈A(s) q*(s,a).


# Monte Carlo Methods 




––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

<br>

![localImage](images/policy.png)

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
