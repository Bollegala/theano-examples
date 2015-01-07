
"""
Demonstrates how to implement binary logistic regression in Theano.

Danushka Bollegala
6th Jan 2015.
"""


import theano
from theano import tensor as T
import sys

import numpy
from svmlight_loader import load_svmlight_file


class LogisticRegression():
    """
    Performs binary logistic regression using cross-entropy error.
    """

    def __init__(self, x, N, D):
        """ 
        Initialize the cost function and gradient for logistic regression

        :type x: theano.tensor.vector
        :param x: symbolic variables that describes the input 

        :type N: int 
        :param N: total number of train instances

        :type D: int
        :param N: dimensionality of the feature space
        """
        # Create a one dimensional tensor (i.e. a vector) for the weight vector.
        # borrow=True does not perform a deep copy of the variable and is faster.
        self.w = theano.shared(value=numpy.zeros(D, dtype=theano.config.floatX), name='w', borrow=True)

        # Initialise the bias
        self.b = theano.shared(value=numpy.float(0), name='b')

        # Symbolic definition of the logistic sigmoid function
        self.p_y_given_x = T.nnet.sigmoid(T.dot(x, self.w) + self.b)

        # Symbolic definition of how to predict the class
        self.y_pred = (T.sgn(self.p_y_given_x - 0.5) + 1) / 2

        # Parameters of the model
        self.params = [self.w, self.b]
        pass


    def negative_log_likelihood(self, y):
        return -T.mean((y * T.log(self.p_y_given_x) + (1 - y) * T.log(1 - self.p_y_given_x)))

    def errors(self, y):
        return (T.mean(T.neq(self.y_pred, y)), T.dot(self.w, self.w), self.b)

    def norm(self):
        return T.norm(self.w)
    pass


def process():
    """ 
    Demonstrates the LogisticRegression class 
    """
    dataset = "rcv1"
    rate = 1000
    epohs = 500
    train_X, train_y = load_svmlight_file("../data/%s/%s.train" % (dataset, dataset))
    N, D = train_X.shape
    train_y = 0.5 * (train_y + numpy.ones(N, dtype=int))
    shared_X = numpy.asarray(train_X.toarray(), dtype=theano.config.floatX)
    shared_y = numpy.asarray(train_y, dtype=theano.config.floatX) 

    x = T.matrix('x')
    y = T.vector('y')
    LR = LogisticRegression(x, N, D)
    cost = LR.negative_log_likelihood(y)
    g_w = T.grad(cost=cost, wrt=LR.w)
    g_b = T.grad(cost=cost, wrt=LR.b)
    updates = [(LR.w, LR.w - rate * g_w), (LR.b, LR.b - rate * g_b)]
    train_model = theano.function(inputs=[x, y], outputs=cost, updates=updates, allow_input_downcast=True)
    err, wval, bval = LR.errors(y)
    test_model = theano.function(inputs=[x, y], outputs=[err, wval, bval], allow_input_downcast=True)
    print "Train instances =", N 
    print "Dimensionality  =", D
    for t in range(epohs):
        likelihood = train_model(shared_X, shared_y)
        err_val, w_val, b_val = test_model(shared_X, shared_y)
        #norm = numpy.dot(wval, wval)
        print "Epoh %d: Likelihood = %f Errors = %f b = %s norm = %s" % (t, likelihood, err_val, str(b_val), str(w_val))

    pass


if __name__ == "__main__":
    process()

