import os, sys, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gym, cv2
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
IMAGE_SIZE = 80
K_ACTIONS = 4
GAMMA = 0.99
TRAIN_EPISODES = 60000
TARGET_UPDATE_INTERVAL = 10000
SAVE_INTERVAL = 200
MAX_BUFFER_SIZE = 500000
MIN_BUFFER_SIZE = 50000
epsilon, EPSILON_MIN = 1.0, 0.1
EPSILON_DELTA = 0.0000001
RMS_LR, RMS_DECAY, RMS_MOMENTUM, RMS_EPSILON = 0.0002, 0.99, 0.0, 1e-6

def dqn_forward(x):

	x = x/255.0
	x = tf.transpose(x, [0, 2, 3, 1])
	
	x = tf.contrib.layers.conv2d(x, 32, 8, 4)
	x = tf.nn.relu(x)
	x = tf.contrib.layers.conv2d(x, 64, 4, 2)
	x = tf.nn.relu(x)
	x = tf.contrib.layers.conv2d(x, 64, 3, 1)
	x = tf.nn.relu(x)
	
	x = tf.contrib.layers.flatten(x)
	x = tf.contrib.layers.fully_connected(x, 512)
	x = tf.contrib.layers.fully_connected(x, K_ACTIONS)

	return x

def preprocess_image(img):
	
	img = img[30:-15,5:-5:,:]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
	
	return gray

if __name__ == '__main__':

	env = gym.envs.make('Breakout-v0')
	
	# agent network
	with tf.variable_scope('agent_network'):
		agent_x = tf.placeholder(tf.float32, shape=(None, 4, IMAGE_SIZE, IMAGE_SIZE), name='x')
		agent_actions_qvals = dqn_forward(agent_x)
	
	# target network
	with tf.variable_scope('target_network'):
		target_x = tf.placeholder(tf.float32, shape=(None, 4, IMAGE_SIZE, IMAGE_SIZE))
		target_actions_qvals = dqn_forward(target_x)

	# compute loss
	target_labels = tf.placeholder(tf.float32, shape=(None,))
	buffer_actions = tf.placeholder(tf.int32, shape=(None,))
	selected_action_values = tf.reduce_max(agent_actions_qvals * tf.one_hot(buffer_actions, K_ACTIONS), axis=1)
	loss = tf.reduce_mean(tf.square(target_labels - selected_action_values))
	train_op = tf.train.RMSPropOptimizer(RMS_LR, RMS_DECAY, RMS_MOMENTUM, RMS_EPSILON).minimize(loss)

	# copy weights from agent network to target network
	agent_weights = [t for t in tf.trainable_variables() if 'agent_network' in t.name]
	agent_weights = sorted(agent_weights, key=lambda v: v.name)
	target_weights = [t for t in tf.trainable_variables() if 'target_network' in t.name]
	target_weights = sorted(target_weights, key=lambda v: v.name)
	update_target_network = []
	for target_weight,agent_weight in zip(target_weights, agent_weights):
		update_target_network.append(tf.assign(target_weight,agent_weight))

	sess = tf.Session()
	saver = tf.train.Saver(max_to_keep=1000)
	sess.run(tf.global_variables_initializer())
	
	# initialize replay buffer
	replay_buffer = []
	observation = env.reset()
	observation = preprocess_image(observation)
	state = np.stack([observation]*4)
	for i in range(MIN_BUFFER_SIZE):

		action = np.random.choice(K_ACTIONS)
		observation, reward, done, _ = env.step(action)
		observation = preprocess_image(observation)
		next_state = np.concatenate([state[1:], np.expand_dims(observation, 0)], axis=0)
		
		replay_buffer.append((state, action, reward, next_state, done))

		if not done:
			state = next_state
		else:
			observation = env.reset()
			observation = preprocess_image(observation)
			state = np.stack([observation]*4)

	# train for number of episodes
	train_iter = 0
	for episode_iter in range(TRAIN_EPISODES):

		if (episode_iter + 1)%SAVE_INTERVAL == 0:
			if not os.path.exists('./models'):
				os.makedirs('./models')
			saver.save(sess, './models/net', global_step=episode_iter+1)
		
		# play single episode, update replay buffer and learn from it
		observation = env.reset()
		observation = preprocess_image(observation)
		state = np.stack([observation]*4)
		total_reward, num_steps = 0, 0
		done = False
		while not done:

			# update target network
			if train_iter % TARGET_UPDATE_INTERVAL == 0:
			  print 'train_iter:', train_iter, '- update_target_network'
			  sess.run(update_target_network)

			# pick an action with respect to epsilon-gredy policy
			if np.random.random() < epsilon:
				action = np.random.choice(K_ACTIONS)
			else:
				actions_qvals = sess.run(agent_actions_qvals, feed_dict={agent_x: [state]})[0]
				action = np.argmax(actions_qvals)
				
			observation, reward, done, _ = env.step(action)
			observation = preprocess_image(observation)
			next_state = np.concatenate([state[1:], np.expand_dims(observation, 0)], axis=0)
			
			# append sample to replay buffer
			if len(replay_buffer) == MAX_BUFFER_SIZE: replay_buffer.pop(0)
			replay_buffer.append((state, action, reward, next_state, done))
			state = next_state
			
			# sample from replay buffer
			samples_batch = random.sample(replay_buffer, BATCH_SIZE) # list of tuples
			states, actions, rewards, next_states, dones = zip(*samples_batch) # tuple of lists

			# pass next states through target network to get q values
			target_actions_qvals_data = sess.run(target_actions_qvals, feed_dict={target_x: next_states})
			target_max_action_qval_data = np.max(target_actions_qvals_data, axis=1)
			# apply Bellman equation to compute target labels: target [for Q(s,a)] = reward + discount * max [over a'] from Q(s', a')
			targets = rewards + np.logical_not(dones).astype(np.float32) * GAMMA * target_max_action_qval_data

			# update agent network by passing current states and forcing Bellman equation with MSE on sampled actions
			sess.run(train_op, feed_dict={agent_x: states, target_labels: targets, buffer_actions: actions})
			
			total_reward = total_reward + reward
			num_steps = num_steps + 1
			train_iter = train_iter + 1
			epsilon = max(epsilon - EPSILON_DELTA, EPSILON_MIN)

		print 'episode iter:', episode_iter, 'reward:', total_reward, 'number of steps:', num_steps
		print 'train_iter:', train_iter, 'epsilon:', epsilon
