
import numpy as np
import torch
import collections


class Agent:

	# Function to initialise the agent
	def __init__(self):
		# Set the episode length
		self.episode_length = 300
		# Reset the total number of steps which the agent has taken
		self.num_steps_taken = 0
		# The state variable stores the latest state of the agent in the environment
		self.state = None
		# The action variable stores the latest action which the agent has applied to the environment
		self.action = None
		# Create the agent's action set
		self.action_set = np.array([0,1,2,3])
		# Set up the agent's Q-network
		self.dqn = DQN(0.001)
		# Set up the agent's experience replay buffer
		self.replaybuffer = ReplayBuffer()

	# Function to check whether the agent has reached the end of an episode
	def has_finished_episode(self):
		if self.num_steps_taken % self.episode_length == 0:
			return True
		else:
			return False

	# Function to get the next action, with current state as the argument
	def get_next_action(self, state):
		# Increase the step count by 1
		self.num_steps_taken += 1
		# Define epsilon such that, the agent becomes more greedy as the number of steps taken increases
		if self.num_steps_taken/1000 >=12:
			epsilon = 1/((self.num_steps_taken/1000)**(0.8))
		elif self.num_steps_taken/1000 < 1:
			epsilon = 1
		else:
			epsilon = 1/((self.num_steps_taken/1000)**(0.45))
		# if self.num_steps_taken/1000 < 1:
		# 	epsilon = 1
		# else:
		# 	epsilon = 1/(self.num_steps_taken/1000)
		# Obtain the index of the discrete action with the highest state-action value
		greedy_move = self.get_greedy_move(state)
		# Set probability vector to be epsilon greedy
		probability = [epsilon/4, epsilon/4, epsilon/4, epsilon/4]
		probability[greedy_move] = (1-epsilon)+(epsilon/4)
		# Choose a move according to the probability vector
		move = np.random.choice(self.action_set, p = probability)
		# Convert the discrete move into continuous action
		action = self._get_continuous_action(move)
		# Store the state; this will be used later, when storing the transition
		self.state = state
		# Store the action as the discrete action; this will be used later, when storing the transition
		self.action = move
		return action

	# Function to set the next state and distance, which resulted from applying action self.action at state self.state
	def set_next_state_and_distance(self, next_state, distance_to_goal):
		# Convert the distance to a reward
		reward = 1 - distance_to_goal
		# Create a transition
		transition = (self.state, self.action, reward, next_state)
		# Put the transition into experience replay buffer
		self.replaybuffer.add_buffer(transition)
		# Update the target network every 1000 steps
		if self.num_steps_taken>0 and self.num_steps_taken%1000 == 0:
			self.dqn._update_target()
		# When the buffer has a sufficient number of transitions, start training
		if len(self.replaybuffer.buffer) >= 500:
			# Sample 500 transitions from the minibatch
			state, action, reward, next_state = self.replaybuffer.sample_buffer(500)
			# Calculate loss
			self.dqn.train_network(state, action, reward, next_state, 0.95)

	# Function to obtain the greedy continuous action for a particular state
	def get_greedy_action(self, state):
		# Send the state into the current network to output the state-action values
		state_tensor = torch.tensor(state, dtype=torch.float32)
		predicted_qvals_tensor = self.dqn.q_network.forward(state_tensor).detach().numpy()
		# Obtain the highest state-action value index
		move = np.argmax(predicted_qvals_tensor)
		# Convert the move into continuous action
		action = self._get_continuous_action(move)
		return action

	# Function to determine continuous action from discrete action
	def _get_continuous_action(self, move):
		if move == 0:
			action = np.array([0, 0.02]).astype(np.float32)
		elif move == 1:
			action = np.array([0.02, 0]).astype(np.float32)
		elif move == 2:
			action = np.array([0, -0.02]).astype(np.float32)
		elif move == 3:
			action = np.array([-0.02, 0]).astype(np.float32)
		return action

	# Function to get the greedy discrete action for a particular state
	def get_greedy_move(self, state):
		# Send the state into the current network to output the state-action values
		state_tensor = torch.tensor(state, dtype=torch.float32)
		predicted_qvals_tensor = self.dqn.q_network.forward(state_tensor).detach().numpy()
		# Obtain the highest state-action value index
		greedy_move = np.argmax(predicted_qvals_tensor)
		return greedy_move



# The Network class inherits the torch.nn.Module class which is a neural network
class Network(torch.nn.Module):

	# The initiation function has 2 arguments: the dimension of the network's input and the dimension of the network's output
	# Network input dimension is 2, representing the state coordinates. The output dimension is 4, representing the actions. 
	def __init__(self, input_dim, output_dim):
		# Call the initiation function of the parent class
		super(Network, self).__init__()
		# Create the network layers
		self.layer_1 = torch.nn.Linear(in_features=input_dim, out_features=500)
		self.layer_2 = torch.nn.Linear(in_features=500, out_features=500)
		self.output_layer = torch.nn.Linear(in_features=500, out_features=output_dim)

	# The function which sends input data through the network and returns the network output
	def forward(self, input):
		layer_1_output = torch.nn.functional.relu(self.layer_1(input))
		layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
		output = self.output_layer(layer_2_output)
		return output



# The DQN class trains the neural network
class DQN:

	# The initiation function has 1 argument, the learning rate.
	def __init__(self, learning_rate):
		# Create a Q-network which predicts the state-action values given a state
		self.q_network = Network(input_dim=2, output_dim=4)
		# Create a target network for double Q-learning
		self.target_network = Network(input_dim=2, output_dim=4)
		# Define the optimiser for updating the network
		self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

	# The function for training the Q-network
	def train_network(self, state, action, reward, next_state, gamma):
		# Set all the gradients stored in the optimiser to zero.
		self.optimiser.zero_grad()
		# Calculate the loss for this transition.
		loss = self._calc_loss(state, action, reward, next_state, gamma)
		# Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
		loss.backward()
		# Take one gradient step to update the Q-network.
		self.optimiser.step()

	# The function for calculating loss, given a minibatch
	def _calc_loss(self, state, action, reward, next_state, gamma):
		# Create the state tensor of the minibatch
		state_tensor = torch.tensor(state, dtype=torch.float32)
		# Create the action tensor of the minibatch
		action_tensor = torch.tensor(action, dtype=torch.int64)
		# Create the immediate reward tensor of the minibatch
		reward_tensor = torch.tensor(reward, dtype=torch.float32)
		# Create the next_state tensor of the minibatch
		next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
		
		# Find the predicted q-values of the network
		predicted_qvals_tensor = self.q_network.forward(state_tensor).gather(dim = 1, index = action_tensor.unsqueeze(-1)).squeeze(-1)
		
		# Find all the predicted state-action values of the next state with the target network
		predicted_next_qvals_target = self.target_network.forward(next_state_tensor)
		# Make sure this does not update the target network
		predicted_next_qvals_target = predicted_next_qvals_target.detach()

		# Find all the predicted state-action values of the next state with the current Q-network
		predicted_next_qvals_current = self.q_network.forward(next_state_tensor)
		# Make sure this does not lead to any update
		predicted_next_qvals_current = predicted_next_qvals_current.detach()
		
		# Obtain a tensor of the maximum predicted state-action values, indexed by target network but obtained value from Q-network
		state_action_vals = predicted_next_qvals_current.gather(dim=1, index = predicted_next_qvals_target.argmax(1).unsqueeze(-1)).squeeze(-1)
		# Calculate mean square loss
		loss = torch.nn.MSELoss()(predicted_qvals_tensor, reward_tensor + gamma*state_action_vals)
		#return the loss for this step
		return loss

	# Function to update the target network with the Q-network
	def _update_target(self):
		self.target_network.load_state_dict(self.q_network.state_dict())



# The ReplayBuffer class implements an experience replay buffer
class ReplayBuffer:

	# The class initiation function 
	def __init__(self):
		# Create a collections.deque
		self.buffer = collections.deque(maxlen = 5000)

	# Add item into the Replay Buffer
	def add_buffer(self, tuple):
		# Append a tuple to the deque
		self.buffer.append(tuple)

	# Randomly sample a minibatch from the Replay Buffer
	def sample_buffer(self, size):
		# Obtain random indices from the deque according to minibatch size
		indices = np.random.choice(len(self.buffer), size, replace = False)
		# Create a list of transitions from the deque indices
		minibatch = []
		for i in indices:
			minibatch.append(self.buffer[i])
		# To obtain the state, action, reward, next_state from the minibatch
		state = []
		action = []
		reward = []
		next_state  = []
		for transition in minibatch:
			state.append(transition[0])
			action.append(transition[1])
			reward.append(transition[2])
			next_state.append(transition[3])
		return state, action, reward, next_state











