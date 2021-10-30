import numpy as np
import matplotlib.pyplot as plt


########################## internal functions for building the decision tree ############################

#calculate the entropy level of a dataset as function input
def entropy(x):
	elements, counts = np.unique(x[:,7], return_counts = True)
	entropy = 0
	for i in range(len(elements)):
		if counts[i]!=0:
			entropy += (-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts))
	return entropy

#calculate information gain from a split, with total, left and right dataset as function input
def info_gain(total, left, right):
	gain = entropy(total) - (len(left[:,7])/(len(left[:,7])+len(right[:,7])))\
	*entropy(left) - (len(right[:,7])/(len(left[:,7])+len(right[:,7])))*entropy(right)
	return gain

#returns the attribute and split value
def find_split(x):
	attr, val, score = -1, -1, -999999
	for j in range(0,7):
		sorted_x = x[x[:,j].argsort()]
		for i in range(1,len(x[:,0])):
			if sorted_x[i,j] != sorted_x[i-1,j]:
				gain = info_gain(sorted_x, sorted_x[:i,:], sorted_x[i:,:])
				if gain > score:
					attr, val, score = j, sorted_x[i,j], gain
	return attr, val

############################# internal functions for decision tree end here #################################



############################# internal functions for creating confusion matrix ###############################

#create balanced training and test datasets
def make_fold10(fold, dataset):
	number = [0,500,1000,1500]
	labels=[]
	test = []
	train =[]
	for no in number:
		labels.append(dataset[no:no+500,:])
	la = labels[0]
	test = la[(fold-1)*50:fold*50, :]
	train = np.delete(la, slice((fold-1)*50, fold*50), 0)
	for i in range (1,4):
		la = labels[i]
		test = np.vstack((test, la[(fold-1)*50:fold*50,:]))
		train = np.vstack((train, np.delete(la, slice((fold-1)*50, fold*50), 0)))
	return train, test

#create balanced training and validation datasets for pruning after make_fold10()
def make_fold9(fold, dataset):
	number = [0, 450, 900, 1350]
	labels = []
	validation = []
	train = []
	for no in number:
		labels.append(dataset[no:no+450,:])
	la = labels[0]
	validation = la[(fold-1)*50:fold*50, :]
	train = np.delete(la, slice((fold-1)*50, fold*50), 0)
	for i in range (1,4):
		la = labels[i]
		validation = np.vstack((validation, la[(fold-1)*50:fold*50,:]))
		train = np.vstack((train, np.delete(la, slice((fold-1)*50, fold*50), 0)))
	return train, validation

#takes 1D array as input to verify that single data point
def verify_onepoint(x,tree):
	if tree['leaf'] == 0:
		if x[tree['attribute']] < tree['value']:
			return verify_onepoint(x, tree['left'])
		else:
			return verify_onepoint(x, tree['right'])
	else:
		if x[7] == tree['value']:
			return x[7]
		else:
			return tree['value']

#create 1 confusion matrix, with row as actual label, column as predicted label
def confusion_matrix(test, tree):
	con_matrix = np.zeros((4,4))
	for row in test:
		a = verify_onepoint(row, tree)
		con_matrix[int(row[7]-1),int(a-1)] += 1
	return con_matrix

############################# internal functions for confuson matrix end here ##################################



##################################### internal functions for pruning ###########################################

#evaluate a node to decide whether to prune or not
def prune_evau(tree,node,test):
	node_ori=node.copy()
	pre_accuracy=evaluation(test,tree)
	l_node=node['left']
	r_node = node['right']
	node['quantity']=l_node['quantity']+r_node['quantity']
	node['depth']=min(l_node['depth'],r_node['depth'])-1
	node['leaf']=1
	node.pop('left')
	node.pop('right')
	node.pop('attribute')
	if(l_node['quantity']<r_node['quantity']):
		node['value']=r_node['value']
	else:
		node['value']=l_node['value']
	prune_accuracy=evaluation(test,tree)
	if(prune_accuracy<=pre_accuracy):
		node.pop('quantity')
		node.pop('depth')
		node['leaf'] = 0
		node['left']=node_ori['left']
		node['right']=node_ori['right']
		node['attribute']=node_ori['attribute']
		node['value']=node_ori['value']
	return

#to prune a tree, input (tree, tree, validation set), output a pruned tree
def prune_tree(tree,node,test):
	if (node['leaf']==1):
		return
	prune_tree(tree,node['left'],test)
	prune_tree(tree,node['right'],test)
	if ((node['left']['leaf'] == 1) and (node['right']['leaf'] == 1)):
		prune_evau(tree,node,test)

################################ internal functions for pruning end here ##########################################



##shuffle dataset within labels without messing up label order from 1 to 4
def within_shuffle(dataset):
	dataset = dataset[dataset[:,7].argsort()]
	number = [0, 500, 1000, 1500]
	labels = []
	for no in number:
		labels.append(dataset[no:no+500,:])
	for la in labels:
		la = np.random.shuffle(la)
	new = labels[0]
	for i in range(3):
		new = np.vstack((new, labels[i+1]))
	return new

#shuffle the whole dataset,for noisy data
def full_shuffle(dataset):
	np.random.shuffle(dataset)
	return

#build the decision tree
def tree_learning(dataset, depth):
	if np.max(dataset[:,7]) == np.min(dataset[:,7]):
		qty = len(dataset[:,7])
		return {'value': dataset[0,7], 'quantity': qty, 'depth':depth, 'leaf':1}
	else:
		attri, valu = find_split(dataset)
		node = {'attribute': attri, 'value': valu, 'leaf':0}
		dataset = dataset[dataset[:,attri].argsort()]
		l_dataset = dataset[dataset[:,attri]<valu,:]
		r_dataset = dataset[dataset[:,attri]>=valu,:]
		depth += 1
		node['left'] = tree_learning(l_dataset, depth)
		node['right']=tree_learning(r_dataset, depth)
	return node

#returns the accuracy of one tree, with one trained tree and test dataset
def evaluation(test, tree):
	correct = 0
	total = 0
	for row in test:
		total += 1
		a = verify_onepoint(row, tree)
		if row[7] == a:
			correct += 1
		else:
			pass
	accuracy = correct/total
	return accuracy

#for evaluation without pruning, get the average result from 10-fold cross validation, with the whole dataset as input
def noprune_avgmatrix(dataset):
	matrices = []
	for i in range(10):
		training, testing = make_fold10(i+1, dataset)
		mytree = tree_learning(training, 0)
		confusion = confusion_matrix(testing, mytree)
		matrices.append(confusion)
	summatrix= matrices[0]
	number = list(range(9))
	for num in number:
		summatrix += matrices[num+1]
	avg_confusion = (1/10)*summatrix
	return avg_confusion

#for evaluation with pruning, get the average of 90 confusion matrices, with the whole dataset as input
def yesprune_avgmatrix(dataset):
	matrices = []
	for i in range(10):
		training, test = make_fold10(i+1, dataset)
		for j in range(9):
			train, validation = make_fold9(j+1, training)
			mytree = tree_learning(train, 0)
			prune_tree(mytree, mytree, validation)
			confusion = confusion_matrix(test, mytree)
			matrices.append(confusion)
	summatrix = matrices[0]
	number = list(range(89))
	for num in number:
		summatrix += matrices[num+1]
	avg_confusion = (1/90)*summatrix
	return avg_confusion

#returns recall, precision and F1 for each class, and the average classification rate 
def get_metrics(con_matrix):
	precision = []
	recall = []
	f1 = []
	for i in range(4):
		prec = con_matrix[i,i]/sum(con_matrix[:,i])
		precision.append(prec)
		reca = con_matrix[i,i]/sum(con_matrix[i,:])
		recall.append(reca)
		f = (2*prec*reca)/(prec+reca)
		f1.append(f)
	total = 0
	good = 0
	for j in range(len(con_matrix[:,0])):
		total += np.sum(con_matrix[:,j])
		good += con_matrix[j,j]
	class_rate = good/total
	return precision, recall, f1, class_rate





################################################ get results ########################################

#load data here
data = np.loadtxt("clean_dataset.txt")
#data = within_shuffle(data)

#create confusion matrix directly from data without pruning 
noprune_matrix = noprune_avgmatrix(data)

#create confusion matrix directly from data with pruning
yesprune_matrix = yesprune_avgmatrix(data)

#get evaluation metrics from the confusion matrix
noprune_metrics = get_metrics(noprune_matrix)
yesprune_metrics = get_metrics(yesprune_matrix)

print(noprune_matrix)
print(noprune_metrics)

print(yesprune_matrix)
print(yesprune_metrics)









############################################### plot diagram  ##########################################
# #traverse through the binary tree to get the total nodes to the left and to the right the current node
# def dfs(tree):
#     if tree['leaf'] == 1:
#         return 1, {'left_cnt': 0, 'right_cnt': 0}
#     node = {'left_cnt': 0, 'right_cnt': 0}
#     node['left_cnt'], node['left'] = dfs(tree['left'])
#     node['right_cnt'], node['right'] = dfs(tree['right'])
#     return node['left_cnt'] + node['right_cnt'] + 1, node

# # get all the edges we need to plot and store the edges in an array with their position
# def get_edge(tree, d, x, y, px, py, edges, indent_x, indent_y):
#     pos_x = (x + d['left_cnt'] + 1) * indent_x
#     if y != py:
#         edges.append([[pos_x, px], [y, py]])
#     if tree['leaf'] != 1:
#         get_edge(tree['left'], d['left'], x, y - indent_y, pos_x, y, edges, indent_x, indent_y)
#         get_edge(tree['right'], d['right'], x + d['left_cnt'] + 1, y - indent_y, pos_x, y, edges, indent_x, indent_y)

# # get all the nodes we need to plot and store the nodes in an array
# def get_node(tree, d, x, y, nodes, indent_x, indent_y):
#     pos_x = (x + d['left_cnt'] + 1) * indent_x
#     if tree['leaf'] == 1:
#         nodes.append([str(tree['value']), pos_x, y])
#     else:
#         get_node(tree['left'], d['left'], x, y - indent_y, nodes, indent_x, indent_y)
#         text = 'x' + str(tree['attribute']) + '<=' + str(tree['value'])
#         nodes.append([text, pos_x, y])
#         get_node(tree['right'], d['right'], x + d['left_cnt'] + 1, y - indent_y, nodes, indent_x, indent_y)

# # plot the diagram, combining edges and nodes
# def plot_tree(tree, name, indent_x, indent_y, fig_width, fig_height):
#     degree = dfs(tree)[1]
#     edges = []
#     nodes = []
#     get_edge(tree, degree, 0, 1, 0, 1, edges, indent_x, indent_y)
#     get_node(tree, degree, 0, 1, nodes, indent_x, indent_y)
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))

#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     ax.axis('off')
#     for node in nodes:
#         ax.text(node[1], node[2], node[0], bbox = props)
#     for edge in edges:
#         ax.plot([edge[0][0], edge[0][1]], [edge[1][0], edge[1][1]])

#     fig.savefig(name)

# clean = np.loadtxt("clean_dataset.txt")
# noisy = np.loadtxt("noisy_dataset.txt")

# clean_tree = tree_learning(clean, 0)
# noisy_tree = tree_learning(noisy, 0)

# plot_tree(clean_tree, 'clean_tree.png', 20, 2.5, 40, 20)
# plot_tree(noisy_tree, 'noisy_tree.png', 120, 7.5, 230, 60)







