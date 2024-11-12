A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

Note
The project environment is similar to, but not identical to the Banana Collector environment on the Unity ML-Agents GitHub page(opens in a new tab).

You are required to work with the environment that we will provide as part of the project.

In particular, your project submission should not use the environment on the ML-Agents GitHub page.

Follow the instructions in the Jupyter notebook below to play the game, as a human agent!

The available controls are:

W - move forward. (Note: when playing the game, the agent will move forward, if you don't select a different action in time, so you can also think of this action as the "do nothing" action.)
S - move backward.
A - turn left.
D - turn right.
Spend a couple of minutes moving around and collecting some yellow bananas. Once you feel like you understand the agent's task, feel free to move on to the next part of the lesson!



