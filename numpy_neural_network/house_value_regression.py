import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Import the mini library from part 1
import part1_nn_lib as p1 


class Regressor():
	def __init__(self, x, neurons = [9,1], activations=["sigmoid", "identity"], batch_size=32, learning_rate=0.0005, loss_fun="mse", shuffle_flag = True, nb_epoch = 1200):
	# You can add any input parameters you need
		# Remember to set them with a default value for LabTS tests
		""" 
		Initialise the model.
		  
		Arguments:
			- x {pd.DataFrame} -- Raw input data of shape 
				(batch_size, input_size), used to compute the size 
				of the network.
			- nb_epoch {int} -- number of epoch to train the network.

		"""


		# Initialise a LabelBinarizer instance to store the parameters for binarization
		self.lb = preprocessing.LabelBinarizer()
		# Store the normalisation parameters for x
		self.normalising_min = 666
		self.normalising_max = 999
		# Create a copy of the data for initialising the Regressor
		x_init = x.copy()
		# Process the data for initialising the Regressor
		X, _ = self._preprocessor(x_init, y=None, training = True)

		# Attributes of the neural network
		self.input_size = X.shape[1]
		self.output_size = 1
		self.nb_epoch = nb_epoch
		self.neurons = neurons
		self.activations = activations
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.loss_fun = loss_fun
		self.shuffle_flag = shuffle_flag
		# Create an instance of the Trainer class
		self.trainer = 888

		return


	def _preprocessor(self, x, y = None, training = False):
		""" 
		Preprocess input of the network.
		  
		Arguments:
			- x {pd.DataFrame} -- Raw input array of shape 
				(batch_size, input_size).
			- y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
			- training {boolean} -- Boolean indicating if we are training or 
				testing the model.

		Returns:
			- {torch.tensor} -- Preprocessed input array of size 
				(batch_size, input_size).
			- {torch.tensor} -- Preprocessed target array of size 
				(batch_size, 1).

		"""


		# Return preprocessed x and y, return None for y if it was None
		if y is None:
			x_copy = x.copy()
			# Fill the missing values of x
			x_copy.fillna(x_copy.mean(), inplace=True)
			# Fit the textual data, store the binarizing parameters in the binarizer
			# Transform textual data into binary values
			dummy = self.lb.fit_transform(x_copy.loc[:,"ocean_proximity"])
			# Drop the textual column
			x_dropped = x_copy.drop("ocean_proximity", axis = 1, inplace = False)
			# Turn X into a numpy ndarray
			x_array = x_dropped.to_numpy(dtype=float)
			# Hstack X with the binary values
			xp = np.hstack((x_array, dummy))

			# Create an instance of the Preprocessor with training x
			preproc = p1.Preprocessor(xp)
			# Save the normalising parameters
			self.normalising_min = preproc._min
			self.normalising_max = preproc._max
			# Apply the normalisation to X
			xp = preproc.apply(xp)

		# Return preprocessed x and y, when y is not None
		else:
			x_copy = x.copy()
			y_copy = y.copy()
			# Fill missing values with the average value of the column
			x_copy.fillna(x_copy.mean(), inplace=True)
			y_copy.fillna(y_copy.mean(), inplace=True)

			# If data is training data, calculate the normalising and binarizing parameters
			if training:
				# Fit the textual data, store the binarizing parameters in the binarizer
				# Transform textual data into binary values
				dummy = self.lb.fit_transform(x_copy.loc[:,"ocean_proximity"])
				# Drop the textual column
				x_dropped = x_copy.drop("ocean_proximity", axis = 1, inplace = False)
				# Turn X into a numpy ndarray
				x_array = x_dropped.to_numpy(dtype=float)
				# Hstack X with the binary values
				xp = np.hstack((x_array, dummy))
				# Turn Y into a numpy ndarray
				yp = y_copy.to_numpy(dtype = float)
				
				# Create an instance of the Preprocessor with training x
				preproc = p1.Preprocessor(xp)
				# Save the normalising parameters
				self.normalising_min = preproc._min
				self.normalising_max = preproc._max
				# Apply the normalisation to X
				xp = preproc.apply(xp)
			
			# If data is test data, process it with the saved parameters
			else:
				# Transform textual data into binary values with saved parameters
				dummy = self.lb.transform(x_copy.loc[:,"ocean_proximity"])
				# Drop the textual column
				x_dropped = x_copy.drop("ocean_proximity", axis = 1, inplace = False)
				# Turn X into a numpy array
				x_array = x_dropped.to_numpy(dtype = float)
				# Hstack X with binary values
				xp = np.hstack((x_array, dummy))
				# Turn Y into a numpy ndarray
				yp = y_copy.to_numpy(dtype = float)
				# Apply normalisation to X with saved parameters
				xp = (xp - self.normalising_min) / (self.normalising_max - self.normalising_min)

		# As data values are large, divide y data by 1000 to avoid overflow
		return xp, (yp / 1000 if not(y is None) else None)


		
	def fit(self, x, y):
		"""
		Regressor training function

		Arguments:
			- x {pd.DataFrame} -- Raw input array of shape 
				(batch_size, input_size).
			- y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

		Returns:
			self {Regressor} -- Trained model.

		"""


		# Process the training X and Y
		X, Y = self._preprocessor(x, y, training = True) # Do not forget
		# Create a neural network
		net = p1.MultiLayerNetwork(self.input_size, self.neurons, self.activations)
		# Create a trainer
		trainer = p1.Trainer(
			network=net, 
			batch_size=self.batch_size, 
			nb_epoch=self.nb_epoch,
			learning_rate=self.learning_rate,
			loss_fun=self.loss_fun,
			shuffle_flag=self.shuffle_flag
		)
		# Train the network
		trainer.train(X, Y)
		# Save the trainer as an instance of the Regressor
		self.trainer = trainer

		return self


			
	def predict(self, x):
		"""
		Ouput the value corresponding to an input x.

		Arguments:
			x {pd.DataFrame} -- Raw input array of shape 
				(batch_size, input_size).

		Returns:
			{np.darray} -- Predicted value for the given input (batch_size, 1).

		"""



		# Process data
		X, _ = self._preprocessor(x, training = False) # Do not forget
		# Forward the data in the network
		y_predict = self.trainer.network.forward(X)
		# As data is divided by 1000 during training, now multiply by 1000 for prediction
		return y_predict * 1000
		# pass


	def score(self, x, y):
		"""
		Function to evaluate the model accuracy on a validation dataset.

		Arguments:
			- x {pd.DataFrame} -- Raw input array of shape 
				(batch_size, input_size).
			- y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

		Returns:
			{float} -- Quantification of the efficiency of the model.

		"""


		# Obtain the predicted y values
		X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
		y_predict = self.trainer.network.forward(X)

		# calculate R2 score value
		error = 0
		v = 1
		avg = np.average(Y)
		for i in range(y_predict.shape[0]):
			error += (Y[i] - y_predict[i]) ** 2
			v += (Y[i] - avg) ** 2
		print("R2 value: {}\n".format(1 - error / v))

		print("RMSE value: {}\n".format((error/y_predict.shape[0])**(0.5)))

		# calculate the relative absolute error
		abs_error = 0 
		abs_v = 1
		for j in range(y_predict.shape[0]):
			abs_error += abs(Y[j] - y_predict[j])
			abs_v += abs(Y[i] - avg)
		print("MAE value: {}\n".format(abs_error/y_predict.shape[0]))

		# Return the R2 score
		return (1 - error / v)	



def save_regressor(trained_model): 
	""" 
	Utility function to save the trained regressor model in part2_model.pickle.
	"""
	# If you alter this, make sure it works in tandem with load_regressor
	with open('part2_model.pickle', 'wb') as target:
		pickle.dump(trained_model, target)
	print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
	""" 
	Utility function to load the trained regressor model in part2_model.pickle.
	"""
	# If you alter this, make sure it works in tandem with save_regressor
	with open('part2_model.pickle', 'rb') as target:
		trained_model = pickle.load(target)
	print("\nLoaded model in part2_model.pickle\n")
	return trained_model



def RegressorHyperParameterSearch(x_data, y_data, option, hp_range): 
	# Ensure to add whatever inputs you deem necessary to this function
	"""
	Performs a hyper-parameter for fine-tuning the regressor implemented 
	in the Regressor class.

	Arguments:
		Add whatever inputs you need.
		
	Returns:
		The function should return your optimised hyper-parameters. 

	"""

	
	# Create the variable for best R2 score, and a list for all the scores
	best_score = -999
	best_hp = hp_range[0]
	hp_score=[]

	# Create an index for 5-fold cross validation
	split_idx = int(0.2 * len(x_data))

	# Generate a model for every value of a hyperparameter
	for hp_test in hp_range:
		score_set = []

		# split into training set and validation set
		for i in range(0, int(len(x_data) * 0.8), split_idx):
			x_val = x_data.iloc[i:i + split_idx, :].copy()
			y_val = y_data.iloc[i:i + split_idx, :].copy()
			x_train = pd.concat([x_data.iloc[0:i, :].copy(), x_data.iloc[i + split_idx:len(x_data), :]].copy(),
								ignore_index=True)
			y_train = pd.concat([y_data.iloc[0:i, :].copy(), y_data.iloc[i + split_idx:len(x_data), :]].copy(),
								ignore_index=True)

			# Generate the models
			if option == "nb_epoch":
				regressor = Regressor(x_data, nb_epoch=hp_test)
			if option == "neurons":
				regressor = Regressor(x_data, neurons=hp_test)
			if option == "batch_size":
				regressor = Regressor(x_data, batch_size=hp_test)
			if option == "learning_rate":
				regressor = Regressor(x_data, learning_rate=hp_test)
			if option == "activations":
				regressor = Regressor(x_data, activations=hp_test)

			# Use training data to train the network
			regressor.fit(x_train, y_train)

			# Evaluate the model
			score = regressor.score(x_val, y_val)
			score_set.append(score)

		# update hyper_param
		if np.mean(score_set) > best_score:
			best_score = np.mean(score_set)
			best_hp = hp_test
		hp_score.append(np.mean(score_set))
		print("\n{}:{} score: {}\n".format(option, hp_test, np.mean(score_set)))
	
	# Print result
	print(best_hp)
	print(hp_score)
	# Return the chosen hyper parameters
	return best_hp  
	



def example_main():

	output_label = "median_house_value"
	# Use pandas to read CSV data as it contains various object types
	# Feel free to use another CSV reader tool
	# But remember that LabTS tests take Pandas Dataframe as inputs
	data = pd.read_csv("housing.csv")

	# Split 1/5 of data as the held-out test set
	length = int(len(data)*0.2)
	data_test = data.loc[0:length, :].copy()
	data_train = data.loc[length:,:].copy()


	# Split training data into input and output
	x_train = data_train.loc[:, data.columns != output_label]
	y_train = data_train.loc[:, [output_label]]

	# Split test data into input and output
	x_test = data_test.loc[:, data.columns != output_label]
	y_test = data_test.loc[:, [output_label]]

	# Training
	# This example trains on the whole available dataset. 
	# You probably want to separate some held-out data 
	# to make sure the model isn't overfitting

	# Use initiation data to initiate the regressor
	regressor = Regressor(x_train, nb_epoch = 1200)
	# Use training data to train the network
	regressor.fit(x_train, y_train)
	# Save the model
	save_regressor(regressor)

	# Evaluate the model
	error = regressor.score(x_test, y_test)
	print("\nRegressor error: {}\n".format(error))

	# Fine tune the number of neurons
	# best_neurons = RegressorHyperParameterSearch(x_train, y_train, 'neurons',
	# 											 [[3, 1], [5, 1], [7, 1], [9, 1], [11, 1], [13, 1]])

	# # fine tune batchsize
	# best_batch_size = RegressorHyperParameterSearch(x_train, y_train, 'batch_size', [16, 32, 64, 128, 256, 512, 1024, 2048])

	# # fine tune learning rate
	# best_learning_rate = RegressorHyperParameterSearch(x_train, y_train, 'learning_rate',
	# 												   [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

	# # fine tune the number of episodes
	# best_nb_epoch = RegressorHyperParameterSearch(x_train, y_train, 'nb_epoch', [300, 500, 800, 1000, 1200, 1500])

	# # fine tune activation function
	# best_activations = RegressorHyperParameterSearch(x_train, y_train, 'activations',
	# 												 [["leaky relu", "identity"], ["tanh", "identity"],
	# 												  ["sigmoid", "identity"]])

if __name__ == "__main__":
	example_main()

