import numpy as np

#Writing the functions for finding gradient

#gradient function is derived using the usual chain rules

a = np.array([0,1])
b = np.array([-2, 1])
B = np.array([[4,-2],[-2,4]])

c = np.array([-3/5, -2/5])
C = np.array([[3,-2],[-2,3]])


def grad_f1(x):
	f1_grad = 2*(x-c).dot(C)
	return f1_grad


def grad_f2(x):
	u = (x[0]-b[0])**2 + (x[1]-b[1])**2
	u_vec = np.array([-2*np.sin(u)*(x[0]-b[0]), -2*np.sin(u)*(x[1]-b[1])])
	f2_grad = u_vec + 2*(x-a).dot(B)
	return f2_grad


def grad_f3(x):
	u = (x[0]-a[0])**2 + (x[1]-a[1])**2
	v = (x-b).dot(B.dot(x-b))
	f3_grad = -(np.exp(-u)*(-2)*(x-a) + np.exp(-v)*(-2)*(x-b).dot(B) \
		- (1/10)*(1/ ((x[0]**2)*0.01 + (x[1]**2)*0.01 + 1/10000) )*((1/50)*x) )
	return f3_grad




#gradient descent
def grad_descent2(x, max_iter, step_size):
	iteration = 0
	x1_f2=np.array([])
	x2_f2=np.array([])
	while iteration < max_iter:
		x1_f2 = np.append(x1_f2,x[0])
		x2_f2 = np.append(x2_f2,x[1])
		x = x - step_size*grad_f2(x)
		iteration +=1
	return x1_f2, x2_f2


#gradient descent
def grad_descent3(x, max_iter, step_size):
	iteration = 0
	x1_f3 = np.array([])
	x2_f3 = np.array([])
	while iteration < max_iter:
		x1_f3 = np.append(x1_f3,x[0])
		x2_f3 = np.append(x2_f3,x[1])
		x = x - step_size*grad_f3(x)
		iteration +=1
	return x1_f3, x2_f3


x=np.array([0.3,0])
print(grad_descent2(x,50,0.05))





