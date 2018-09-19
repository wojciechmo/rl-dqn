# Deep Q-Learning

Teach RL agent how to play Atari Breakout with Deep Q-Learning and TensorFlow.

<img src="https://github.com/WojciechMormul/rl-dqn/blob/master/imgs/games.png" width="800">

Deep Q-Learning uses neural network as Q-value function approximator. It's also stuffed with various stability tricks. 

Search for Q-value function which satisfies Bellman equation: Q(s,a) = r + gamma * max[over a']Q(s',a'). Find optimal Q-value function recursively. During training force Bellman equation on sampled batches from replay buffer. No supervision available. Need to come up with labels G(s,a) for each (state, action, reward, next_state) training example.

Deep Q-Learning sketch:

<img src="https://github.com/WojciechMormul/rl-dqn/blob/master/imgs/deep%20q-learning.png" width="400">

Deep Q-Network architecture:

<img src="https://github.com/WojciechMormul/rl-dqn/blob/master/imgs/deep%20q-network.png" width="700">

Use experience replay buffer to decorrelate training data by taking different parts from different trajectories instead of consecutive samples from single episode. Store tuples (s,a,r,s') using epsilon-greedy policy with respect to agent network. Update agent model using batches sampled from replay buffer.

Use target network to remove targets dependency on agent network parameters. Target network has the same architecture as agent network but different set of parameters. Copy agent network parameters to target network once in a while.

State is defined as 4 consecutive frames.
