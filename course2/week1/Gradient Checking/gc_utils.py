import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        #np.reshape(parameters[key], (-1,1))这句话的意思是把parameters[key]这个矩阵展开为只有一列，不知道有多少行的向量，也就是说展开成一个列向量，-1表示行数未知
        #z.reshape(-1, 2)就是把z这个矩阵展开为行数未知，两列的矩阵，将一行一行多的数往下铺展

        keys = keys + [key]*new_vector.shape[0]
        #keys是一个空列表，主要是为了区别最后把所有矩阵都变成一个列向量了，来区分到底哪一段属于哪个参数的，所以给列向量的每个元素都加一个标签说明这是哪个参数的，方便在矩向量转化为字典时，能按照对应的形式进行转化为字典
        #[key]表示["W1"]即是一个列表，列表*5得到一个新的列表，里面有5个相同的元素，而字符串乘以5则得到一个更长的有5个该字符串堆叠成的字符串
      	"""
      	keys=[]
		for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
   		 	keys = keys + [key]*5
		print(keys)
		结果为['W1', 'W1', 'W1', 'W1', 'W1', 'b1', 'b1', 'b1', 'b1', 'b1', 'W2', 'W2', 'W2', 'W2', 'W2', 'b2', 'b2', 'b2', 'b2', 'b2', 'W3', 'W3', 'W3', 'W3', 'W3', 'b3', 'b3', 'b3', 'b3', 'b3']
		即一个更长的列表
      	"""

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:20].reshape((5,4))
    parameters["b1"] = theta[20:25].reshape((5,1))
    parameters["W2"] = theta[25:40].reshape((3,5))
    parameters["b2"] = theta[40:43].reshape((3,1))
    parameters["W3"] = theta[43:46].reshape((1,3))
    parameters["b3"] = theta[46:47].reshape((1,1))

    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta