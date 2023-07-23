import numpy as np

# Theano
import theano
import theano.tensor as tensor

tensor5 = tensor.TensorType(theano.config.floatX, (False,) * 5)