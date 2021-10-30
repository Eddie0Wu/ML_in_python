import numpy as np
import matplotlib.pyplot as plt


###################################### methods #########################################
#generate the polynomial basis, with order and scalar x as input
def polynom_basis(order, x):
	haha = np.zeros((order+1,))
	for i in range(order+1):
		haha[i] = x**i
	return haha


#generate the trigonometric basis, with order and scalar x as input
def trig_basis(order,x):
	haha = np.zeros((2*order+1,))
	for i in range(order+1):
		if i == 0:
			haha[i] = 1
		else:
			haha[2*i-1]=np.sin(2*np.pi*i*x)
			haha[2*i] = np.cos(2*np.pi*i*x)
	return haha


#generate the design matrix. 
#func is basis function, order is basis function order, X is a vector
#basis functions refer to the two functions defined above
def design_matrix(func, order, X):
	haha = []
	for i in range(len(X)):
		row = func(order, X[i])
		haha.append(row)
	huehue = np.array(haha)
	return huehue


#calculate omega and sigma square, dmatrix is design matrix, y is a vector of data
def find_omega_sigma(dmatrix, y):
	phitrans = dmatrix.transpose()
	phiphi = phitrans.dot(dmatrix)
	phiphiinv = np.linalg.inv(phiphi)
	#get omega
	omega = phiphiinv.dot(phitrans.dot(y))

	#get sigma square
	point = y-dmatrix.dot(omega)
	pointtrans = point.transpose()
	sigmasq = (1/len(y))*(pointtrans.dot(point))
	return omega, sigmasq


################################## end of methods ########################################





#generate data
N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N,1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)



########### part a ##############

plt.figure()
orders = [0,1,2,3,11]

#plot the curve for each order
for order in orders:

	#limit the domain of order 11 because it gets too large 
	#and makes the graph unreadable
	if order != 11:
		dmatrix = design_matrix(polynom_basis,order,X)
		omega, sigmasq = find_omega_sigma(dmatrix, Y)
		x_val = np.linspace(-0.05,1.0,100)
		y_val = []
		for value in x_val:
			y = (omega.transpose()).dot(polynom_basis(order,value))
			y_val.append(y)
		plt.plot(x_val,y_val, label = "Order {}".format(order))
	
	#set the domain for the other orders
	elif order == 11:
		dmatrix = design_matrix(polynom_basis,order,X)
		omega, sigmasq = find_omega_sigma(dmatrix, Y)
		x_val = np.linspace(-0.05,0.95,100)
		y_val = []
		for value in x_val:
			y = (omega.transpose()).dot(polynom_basis(order,value))
			y_val.append(y)
		plt.plot(x_val,y_val, label = "Order {}".format(order))

#plot graph
plt.scatter(X,Y, label="Data")
plt.xticks(np.arange(-0.2, 1.3, 0.2))
plt.legend()
plt.savefig("part_a", dpi = 600)





########### part b############

plt.figure()
orders = [1,11]

#plot the curve for each order
for order in orders:
	dmatrix = design_matrix(trig_basis,order,X)
	omega, sigmasq = find_omega_sigma(dmatrix, Y)
	x_val = np.linspace(-1,1.2,200)
	y_val = []
	for value in x_val:
		y = (omega.transpose()).dot(trig_basis(order,value))
		y_val.append(y)
	plt.plot(x_val,y_val, label = "Order {}".format(order))


#plot graph
plt.scatter(X,Y, label="Data")
plt.xticks(np.arange(-1, 1.3, 0.2))
plt.legend()
plt.savefig("part_b", dpi = 600)





############ part c ################

orders = list(range(11))

avg_error = []
avg_sigmasq = []

for order in orders:
	error = []
	sigmasq = []

	#carry out the leave-one-out cross validation 
	#to find the error and sigmasq for each fold
	for i in range(len(X)):
		#create folds here 
		test_x = X[i].copy()
		test_y = Y[i].copy()
		Xx = X.copy()
		Yy = Y.copy()
		Xx = np.delete(Xx, [i])
		Yy = np.delete(Yy, [i])
		
		#get the parameters and predicted error
		dmatrix = design_matrix(trig_basis, order, Xx)
		omega, sigma = find_omega_sigma(dmatrix, Yy)
		sigmasq.append(sigma)
		y = (omega.transpose()).dot(trig_basis(order, test_x))
		err = (test_y-y)**2
		error.append(err)

	#get the average error and sigmasq
	erro = sum(error)/len(error)
	sigmaa = sum(sigmasq)/len(sigmasq)

	#as error is still in the format of array, turn it into a float
	haha = erro.tolist()
	errr = haha[0]

	#append the average values
	avg_error.append(errr)
	avg_sigmasq.append(sigmaa)

#plot graphs
plt.figure()
plt.plot(orders, avg_error, label = "Average squared test error")
plt.scatter(orders, avg_error)
plt.plot(orders, avg_sigmasq, label = "Sigma square to max likelihood")
plt.scatter(orders, avg_sigmasq)
plt.xlabel("Number of order")
plt.legend()
plt.savefig("part_c", dpi = 600)













