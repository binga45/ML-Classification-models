import nn
import math
import random
import numpy as np
import itertools

class LogisticRegressionModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new LogisticRegressionModel instance.

        A logistic regressor classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        Initialize self.w and self.alpha here
        self.alpha = *some number* (optional)
        self.w = []
        """
        "*** YOUR CODE HERE ***"
        random.seed(17)
        self.alpha = 0.0031
        print(dimensions)
        #self.w = [random.randrange(0, 1, 1) for i in range(dimensions)] 
        self.w = [random.random() for i in range(dimensions)] 
        #print
        #random.sample(range(0,1), dimensions)

    def get_weights(self):
        """
        Return a list of weights with the current weights of the regression.
        """
        return self.w

    def DotProduct(self, w, x):
        """
        Computes the dot product of two lists
        Returns a single number
        """
        "*** YOUR CODE HERE ***"
        if(len(w) != len(x)):
            return 0
        return sum(j[0]*j[1] for j in zip(w,x))

    def sigmoid(self, x):
        """
        compute the logistic function of the input x (some number)
        returns a single number
        """
        "*** YOUR CODE HERE ***"
        return 1/(1+math.exp(-self.DotProduct(self.w,x)))

    def run(self, x):
        """
        Calculates the probability assigned by the logistic regression to a data point x.

        Inputs:
            x: a list with shape (1 x dimensions)
        Returns: a single number (the probability)
        """
        "*** YOUR CODE HERE ***"
        return self.sigmoid(x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if(self.run(x) >= 0.5): 
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the logistic regression until convergence (this will require at least two loops).
        Use the following internal loop stucture

        for x,y in dataset.iterate_once(1):
            x = nn.as_vector(x)
            y = nn.as_scalar(y)
            ...

        """
        "*** YOUR CODE HERE ***"
        loss = float('-inf')
        while loss != 0.0:
            loss = 0.0
            for x,y in dataset.iterate_once(1):  
                x = nn.as_vector(x)
                y = nn.as_scalar(y)
                if y != self.get_prediction(x):
                    loss += abs(y-self.get_prediction(x))
                    for i in range(len(self.w)):
                        self.w[i] += (self.alpha*(y-self.run(x))*self.run(x)*(1-self.run(x))*x[i])
        



        

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        We did this for you. Nothing for you to do here.
        """
        self.w = nn.Parameter(1, dimensions)
        self.alpha = 0.1

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        #return np.array([sum(j[0]*j[1] for j in zip(self.w,x))])
        return nn.DotProduct(self.w,x)
        #return np.dot(x,nn.constant(self.w))


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        a = 1 if nn.as_scalar(self.run(x)) >=0 else -1
        return a


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        loss = float('-inf')
        while loss != 0.0:
            loss = 0.0
            #batch size didn't work so using 1
            for x,y in dataset.iterate_once(1):
                y = nn.as_scalar(y)
                temp = self.get_prediction(x)
                if temp != y:
                    loss += abs(y-temp)
                    #h = 1 if temp >=0 else -1
                    parameter = self.w #since self.w is already a Perceptron instance.
                    print(parameter)
                    #print('shape: ',x)
                    #parameter.update(x, self.alpha*(y-h))
                    parameter.update(x, y)


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self, dimensions):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #since equivalent to single neuron
        self.w = nn.Parameter(dimensions,1)
        self.b = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #To get a node with (batch_size x num_outputfeatures)
        node = nn.Linear(x, self.w)
        return nn.AddBias(node, self.b)
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = float('inf')
        print(type(dataset.y))
        print(type(dataset.x))
        while loss > 0.137:
            for x,y in dataset.iterate_once(40):
                loss_temp= self.get_loss(x,y)
                w_g, b_g = nn.gradients(loss_temp,[self.w, self.b])
                self.b.update(b_g, -0.00729)
                self.w.update(w_g, -0.00729)
            loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))
            #print(type(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))))
           # loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), dataset.y))




    def closedFormSolution(self, X, Y):
        """
        Compute the closed form solution for the 2D case
        Input: X,Y are lists
        Output: b0 and b1 where y = b1*x + b0
        """
        "*** YOUR CODE HERE ***"
        X1 = np.array(X)
        Y1 = np.array(Y)
        print(X1.shape)
        print(Y1.shape)
        dotproduct = 0
        if(len(X) == len(Y)):
            dotproduct =  np.sum(X1*Y1)
        #print('dotproduct: ', dotproduct)
        prod = np.sum(X1)*np.sum(Y1)
        denom = len(X)*np.sum(X1*X1) - np.sum(X1)**2
        b1 = (len(X)*dotproduct - prod)/denom
        #print('b1: ',b1)
        b0 = (np.sum(Y1)-b1*np.sum(X1))/len(X1)
        return b0, b1

class PolyRegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self, order):
        # Initialize your model parameters here
        """
        initialize the order of the polynomial, as well as two parameter nodes for weights and bias
        """
        "*** YOUR CODE HERE ***"
        self.order = order
        self.w = nn.Parameter(order, 1)
        self.b = nn.Parameter(1,1)
        self.alpha = 0.0005

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        node = nn.Linear(self.computePolyFeatures(x), self.w)
        return nn.AddBias(node, self.b)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x),y)

    def computePolyFeatures(self, point):
        """
        Compute the polynomial features you need from the input x
        NOTE: you will need to unpack the x since it is wrapped in an object
        thus, use the following function call to get the contents of x as a list:
        point_list = nn.as_vector(point)
        Once you do that, create a list of the form (for batch size of n): [[x11, x12, ...], [x21, x22, ...], ..., [xn1, xn2, ...]]
        Once this is done, then use the following code to convert it back into the object
        nn.Constant(nn.list_to_arr(new_point_list))
        Input: a node with shape (batch_size x 1)
        Output: an nn.Constant object with shape (batch_size x n) where n is the number of features generated from point (input)
        """
        "*** YOUR CODE HERE ***"
        point_list = nn.as_vector(point)
        new_point_list = []
        for point in point_list:
            lst = [point]
            temp = list(itertools.chain.from_iterable(itertools.repeat(x, self.order) for x in lst))
            #temp = np.repeat([point],self.order).tolist()
            new_point_list.append(np.power(temp, range(1,self.order+1)).tolist())
            #new_point_list.append(1)
        print(new_point_list)
        return nn.Constant(nn.list_to_arr(new_point_list))

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = float('inf')
        print('size',len(dataset.x))
        print(type(dataset.y))
        print(type(dataset.x))
        while loss > 0.1:
            loss = 0.0
            for x,y in dataset.iterate_once(10):
                loss_temp= self.get_loss(x,y)
                w_g, b_g = nn.gradients(loss_temp,[self.w, self.b])
                self.b.update(b_g, -self.alpha)
                self.w.update(w_g, -self.alpha)
                #loss += nn.as_scalar(loss_temp)
            loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))
        print('size',len(dataset.x))
        
        


class FashionClassificationModel(object):
    """
    A model for fashion clothing classification using the MNIST dataset.

    Each clothing item is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.alpha = 0.00145
        self.w1 = nn.Parameter(784,256)
        #self.w2 = nn.Parameter(512,256)
        self.w2 = nn.Parameter(256,128)
        self.w3 = nn.Parameter(128,64)
        self.w4 = nn.Parameter(64,10)
        self.b1 = nn.Parameter(1,256)
        self.b2 = nn.Parameter(1,128)
        self.b3 = nn.Parameter(1,64)
        self.b4 = nn.Parameter(1,10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        node = nn.AddBias(nn.Linear(x,self.w1), self.b1)

        node = nn.AddBias(nn.Linear(nn.ReLU(node),self.w2), self.b2)

        node = nn.AddBias(nn.Linear(nn.ReLU(node),self.w3), self.b3)

        node = nn.AddBias(nn.Linear(nn.ReLU(node),self.w4), self.b4)

        return node

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        
        return nn.SoftmaxLoss(self.run(x),y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = float('inf')
        print('size',len(dataset.x))
        print(type(dataset.y))
        print(type(dataset.x))
        while loss > 0.1: 
            previous_loss = loss
            loss = 0.0
            for x,y in dataset.iterate_once(100):
                loss_temp= self.get_loss(x,y)
                w1_g,w2_g,w3_g, w4_g, b1_g,b2_g, b3_g, b4_g = nn.gradients(loss_temp,[self.w1, self.w2,self.w3 , self.w4, self.b1, self.b2, self.b3, self.b4])
                # w1_g,w2_g, b1_g,b2_g = nn.gradients(loss_temp,[self.w1, self.w2, self.b1, self.b2])
                self.b1.update(b1_g, -self.alpha)
                self.b2.update(b2_g, -self.alpha)
                self.b3.update(b3_g, -self.alpha)
                self.b4.update(b4_g, -self.alpha)
                self.w1.update(w1_g, -self.alpha)
                self.w2.update(w2_g, -self.alpha)
                self.w3.update(w3_g, -self.alpha)
                self.w4.update(w4_g, -self.alpha)
                #loss += nn.as_scalar(loss_temp)
            loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))
            if abs(previous_loss-loss) < 0.005:
                print("prevloss: ", previous_loss)
                print("currentloss: ", loss)
                print("valaccuracy: ", dataset.get_validation_accuracy())
                break
            if dataset.get_validation_accuracy()>0.82:
                break
        print('size',len(dataset.x))

