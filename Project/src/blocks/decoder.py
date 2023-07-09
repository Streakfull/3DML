import numpy as np

# Theano
import theano
import theano.tensor as tensor

from models.net import tensor5
from lib.layers import Unpool3DLayer, \
    LeakyReLU, SoftmaxWithLoss3D, Conv3DLayer, \
    AddLayer, get_trainable_params


class Decoder:

    def __init__(self, x):

        # (multi_views, self.batch_size, 3, self.img_h, self.img_w),
        n_deconvfilter = [128, 128, 128, 64, 32, 2]
        
        def recurrence():
            conv7a = Conv3DLayer(x,(n_deconvfilter[1], 3, 3, 3))
            rect7a = LeakyReLU(conv7a)
            conv7b = Conv3DLayer(rect7a, (n_deconvfilter[1], 3, 3, 3))
            rect7 = LeakyReLU(conv7b)
            res7 = AddLayer(x, rect7)

            unpool8 = Unpool3DLayer(res7)
            conv8a = Conv3DLayer(unpool8, (n_deconvfilter[2], 3, 3, 3))
            rect8a = LeakyReLU(conv8a)
            conv8b = Conv3DLayer(rect8a, (n_deconvfilter[2], 3, 3, 3))
            rect8 = LeakyReLU(conv8b)
            res8 = AddLayer(unpool8, rect8)

            unpool9 = Unpool3DLayer(res8)
            conv9a = Conv3DLayer(unpool9, (n_deconvfilter[3], 3, 3, 3))
            rect9a = LeakyReLU(conv9a)
            conv9b = Conv3DLayer(rect9a, (n_deconvfilter[3], 3, 3, 3))
            rect9 = LeakyReLU(conv9b)

            conv9c = Conv3DLayer(unpool9, (n_deconvfilter[3], 1, 1, 1))
            res9 = AddLayer(conv9c, rect9)

            conv10a = Conv3DLayer(res9, (n_deconvfilter[4], 3, 3, 3))
            rect10a = LeakyReLU(conv10a)
            conv10b = Conv3DLayer(rect10a, (n_deconvfilter[4], 3, 3, 3))
            rect10 = LeakyReLU(conv10b)

            conv10c = Conv3DLayer(rect10a, (n_deconvfilter[4], 3, 3, 3))
            res10 = AddLayer(conv10c, rect10)

            conv11 = Conv3DLayer(res10, (n_deconvfilter[5], 3, 3, 3))
            softmax_loss = SoftmaxWithLoss3D(conv11.output)

            self.loss = softmax_loss.loss(self.y)
            self.error = softmax_loss.error(self.y)
            self.params = get_trainable_params()
            self.output = softmax_loss.prediction()