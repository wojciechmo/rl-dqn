import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import tensorflow as tf
import numpy as np
from train import *

sess = tf.Session()

saver = tf.train.import_meta_graph('./models/net-20000.meta')
saver.restore(sess, './models/net-20000')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name('agent_network/x:0')
y = graph.get_tensor_by_name('agent_network/fully_connected_1/Relu:0')

env = gym.envs.make("Breakout-v0")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videowriter = cv2.VideoWriter('eval.mp4', fourcc, 60.0, (480, 630))

# play episode by sampling over actions from agent q values
observation = env.reset()
state = np.stack([preprocess_image(observation)]*4)
total_reward, num_steps = 0, 0
done = False
while done==False:

	actions_probs = sess.run(y, feed_dict={x: [state]})

	img = cv2.resize(observation, (480, 630), interpolation=cv2.INTER_NEAREST)
	img = img[...,::-1]
	videowriter.write(img) 
	
	action = np.argmax(actions_probs[0])
	observation, reward, done, _ = env.step(action)
	state = np.concatenate([state[1:], np.expand_dims(preprocess_image(observation), 0)], axis=0)
			
	total_reward = total_reward + reward
	num_steps = num_steps + 1

print 'reward:', total_reward, 'number of steps:', num_steps
videowriter.release()
