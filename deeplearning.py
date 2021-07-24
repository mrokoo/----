import numpy as np
from numpy import random
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

print(train_labels.shape)
print(train_images.shape)
# test_images = train_images.reshape((10000, 28*28))
# test_images = train_images.astype('float32')/255
train_data = (train_images,train_labels)
print(train_data[1])
def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))
def sigmoid_prime(z):
	"""sigmoid函数的导数"""
	return sigmoid(z)*(1 - sigmoid(z))
class Network(object):
	def __init__(self, sizes):
		self.num_layers	= len(sizes) ## sizes列表，各层神经元数量。
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]] ## 生成偏置
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] ## 具体zip语法做完后再检查，生成权重。
	def feedforward(self, a):
		"""若a为输入，则返回输出"""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a 
	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
		if test_data: 
			n_test = len(test_data)
		n = len(training_data)
		print(n)
		for j in range(epochs):
			# tuple(random.shuffle(list(training_data)))
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
			print(mini_batches)
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			if test_data:
				print(f"Epoch{j}: {self.evaluate(test_data)}/{n_test}")
			else: 
				print(f"Epoch{j} complete")

	def update_mini_batch(self, mini_batch, eta):
		"""对一个小批量应用梯度下降算法和反向传播算法来更新神经网络的权重和偏置。mini_batch是由若干
		元组(x, y)组成的列表，eta是学习率."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw 
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb 
					for b, nb in zip(self.biases, nabla_b)]	
	def backprop(self, x, y):
		"""返回一个表示代价函数C_x梯度的元组(nabla_b, nabla_w)。nabla_b和nabla_w是一层接一层的
		numpy数组的列表，类似于self.biases和self.weights。"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# 前馈
		activation = x
		activations = [x] # 一层接一层地存放所有激活值
		zs = [] # 一层接一层地存放所有z向量
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# 反向传播
		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		"""注意，下面循环中的变量l和第2章的形式稍有不同。这里l = 1表示最后一层神经元，l = 2则
			表示倒数第二层，以此类推。这是对书中方式的重编号，旨在利用Python列表的负索引功能。"""
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
			return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		"""返回测试输入中神经网络输出正确结果的数目。注意，这里假设神经网络输出的是最后一层有着
		最大激活值的神经元的索引。"""
		test_results = [(np.argmax(self.feedforward(x)), y)
							for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		"""返回关于输出激活值的偏导数的向量。"""
		return (output_activations-y)	


net = Network([784, 30, 10])
net.SGD((train_images, train_labels), 30, 100, 3.0, test_data = (test_images, test_labels))

