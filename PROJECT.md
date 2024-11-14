[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the course GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the course GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.


––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


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
