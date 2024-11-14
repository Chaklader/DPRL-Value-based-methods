# Project 1: Navigation

## Description of the implementation

### Algorithms
In order to solve this challenge, I have explored and implemented some of the cutting-edge deep reinforcement learning algorithms, such as:

* [Deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Double Deep Q-Network](https://arxiv.org/abs/1509.06461)
* [Dueling Q-Network](https://arxiv.org/abs/1511.06581)
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

##### Report

The report details the implementation of various deep reinforcement learning algorithms to solve a navigation challenge. Initially, a Deep Q-Network (DQN) was developed with different architectures, starting with a model of two fully-connected layers with 512 nodes each, which solved the environment in 801 episodes. Through experimentation, a more efficient model with two layers of 128 and 32 nodes was developed, solving the environment in 344 episodes. This model served as a baseline for further comparisons.

The training process for the Deep Q-Learning agent was executed on a CUDA-enabled device, indicating GPU acceleration. During training, a warning was issued regarding the retrieval of the last learning rate from the scheduler, suggesting the use of `get_last_lr()` for accurate information. By Episode 100, the agent achieved an average score of 7.34, with an epsilon value of 0.0270 and a learning rate of approximately 0.000227. By Episode 200, the agent's performance improved, reaching an average score of 10.10, with the epsilon value decreasing to 0.0100 and the learning rate further decaying to approximately 0.000107.

The conclusion highlights that while the DQN architecture was effective, further enhancements could be explored. Future work suggestions include testing different hyperparameter sets, aiming for higher scores, introducing negative rewards to discourage random actions, and implementing other advanced algorithms like A3C, Noisy DQN, and Distributional DQN to potentially improve performance. The results suggest that while the agent is learning and improving its performance, further tuning of hyperparameters or exploration of alternative architectures might be necessary to reach the desired performance level.