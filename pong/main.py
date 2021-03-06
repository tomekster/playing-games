# Source: https://keras.io/examples/rl/deep_q_network_breakout/
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from model import Model 
from tensorflow import keras
import argparse

# Configuration params for the whole setup
seed = 42
gamma = 0.99 # Discount factor for >past< rewards
epsilon = 1.0 # Eps greedy param
epsilon_min = 0.1 # Min eps greedy
epsilon_max = 1.0 # Max eps greedy
epsilon_interval = (
	epsilon_max - epsilon_min
) # Rate at which to reduce chance of random action being taken
batch_size = 32

max_steps_per_episode = 10000

ENV_ID = "BreakoutNoFrameskip-v4"
env = make_atari(ENV_ID)
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

input_shape = (84, 84, 4)
num_actions = 4
model = Model(input_shape, num_actions).build()
model_target = Model(input_shape, num_actions).build()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

parser = argparse.ArgumentParser()
parser.add_argument("--frame")
parser.add_argument("--modelpath")
args = parser.parse_args()

if args.frame: frame_count = int(args.frame)
if args.modelpath: 
	model = keras.models.load_model(args.modelpath, compile=False)
	print("Loaded model {}".format(args.modelpath))


# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of framaes for exploration
epsilon_greedy_frames = 1000000.0
# Maximum rerplyaya length
# Note: DThe deepmind paper sugests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to updatet the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

while True:
	state = np.array(env.reset())
	episode_reward = 0

	for timestep in range(1, max_steps_per_episode):
		# env.render() #Adding this line would show the attempts 
		# of the agent in a pop up window
		frame_count += 1

		# Use epsilon-greedy for exploration
		if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
			# Take random action
			action = np.random.choice(num_actions)
		else:
			# Predict action Q-values from environment state
			state_tensor = tf.convert_to_tensor(state)
			state_tensor = tf.expand_dims(state_tensor, 0)
			action_probs = model(state_tensor, training=False)
			# Take best action
			action = tf.argmax(action_probs[0]).numpy()

		# Decay probability of taking random action
		epsilon -= epsilon_interval / epsilon_greedy_frames
		epsilon = max(epsilon, epsilon_min)

		# Apply the sampled action in our environment 
		state_next, reward, done, _ = env.step(action)
		state_next = np.array(state_next)

		episode_reward += reward

		# Save actions and states in replay buffer
		action_history.append(action)
		state_history.append(state)
		state_next_history.append(state_next)
		done_history.append(done)
		rewards_history.append(reward)
		state = state_next

		# Update every fourth frame and once batch size is over 32
		if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
			# Get indices of samples for replay buffers
			indices = np.random.choice(range(len(done_history)), size=batch_size)
			
			# Using list comprehension to sample from replay buffer
			state_sample = np.array([state_history[i] for i in indices])
			state_next_sample = np.array([state_next_history[i] for i in indices])
			rewards_sample = [rewards_history[i] for i in indices]
			action_sample = [action_history[i] for i in indices]
			done_sample = tf.convert_to_tensor(
				[float(done_history[i]) for i in indices]
			)

			# Build the updated Q-values for the sampled future states
			# Use the target model for stability
			future_rewards = model_target.predict(state_next_sample)
			# Q value = reward + discount factor * expected future reward
			updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)


			# If final frame set the last value to -1
			updated_q_values = updated_q_values * (1 - done_sample) - done_sample

			# Create a mask so we only calculate loss on the updated Q-values
			masks = tf.one_hot(action_sample, num_actions)

			with tf.GradientTape() as tape:
				# Train the model on the states and updated Q-values
				q_values = model(state_sample)
				# Apply the masks to the Q-values to get the Q-value for action taken
				q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
				# Calculate loss between new Q-value and old Q-value
				loss = loss_function(updated_q_values, q_action)

			# Backpropagation
			grads = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			
		if frame_count % update_target_network == 0:
			# update the target network with new weights
			model_target.set_weights(model.get_weights())
			# Log details
			template = "running reward: {:.2f} at episode {}, frame count {}"
			print(template.format(running_reward, episode_count, frame_count))
			model.save('saved_model_{}'.format(frame_count))

		# Limit the state and reward history
		if len(rewards_history) > max_memory_length:
			del rewards_history[:1]
			del state_history[:1]
			del state_next_history[:1]
			del action_history[:1]
			del done_history[:1]

		if done:
			break

	# Update running reward to check condition for solving
	episode_reward_history.append(episode_reward)
	if len(episode_reward_history) > 100:
		del episode_reward_history[:1]
	running_reward = np.mean(episode_reward_history)

	episode_count += 1

	if running_reward > 40: # Condition to consider the task solved
		print("Solved at episode {}!".format(episode_count))
		break
