from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import keras
from keras import backend as K, optimizers

from keras.engine import Layer
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import DepthwiseConv2D
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import ReLU
from keras.layers import GaussianDropout
from keras.layers import add
from keras.layers import concatenate
from keras.layers import multiply
from keras.regularizers import l2

from keras.backend import get_session

#############################################################################

def MobileNetV2(input_shape=None,
                re_shape=(2,100,80,1),
                alpha=0.25,
                depth_multiplier=1,
                classes=5,
                weightdecay=0.0002,
                jitter=0.001,
                input_tensor=None):

    """MobileNetv1
    This function defines a MobileNetv1 architectures.
    # Arguments
        inputs: Inuput Tensor, e.g. an image
        alpha: Width Multiplier
        depth_multiplier: Resolution Multiplier
        classes: number of labels
        weightdecay: weight decay for last layer
    # Returns
	five MobileNetv2 model stages."""
    if input_tensor is None:
        img_input = Input(shape=input_shape[0], name='input')
    inputs=img_input
    if jitter != 0:
        img_input_jitter = GaussianDropout(jitter)(img_input)
        shaped = Reshape(re_shape)(img_input_jitter)
    else:
        shaped = Reshape(re_shape)(img_input)
    def _lambda_unstack(x):
        import tensorflow as tf
        return tf.unstack(x,axis=1)
    shaped = Lambda(_lambda_unstack)(shaped)
    img_input1 = shaped[0]
    img_input2 = shaped[1]
    if(re_shape[0] == 4):
        img_input3 = shaped[2]
        img_input4 = shaped[3]
    if(re_shape[0] == 4):
        img_input = [img_input1, img_input2, img_input3, img_input4]
    else:
        img_input  = [img_input1, img_input2]

    branches = []
    names = ['x','y','px','py']
    for i in range(len(img_input)):
        branch = subnet(img_input[i], names[i], alpha)
        branches.append(branch)

    merge = concatenate(branches)

    merge = _inverted_residual_block(merge, 64,  (3, 3), t=6, strides=2, n=4, alpha=alpha, block_id=7, name='merge')
    merge = _inverted_residual_block(merge, 96,  (3, 3), t=6, strides=1, n=3, alpha=alpha, block_id=11, name='merge')
    merge = _inverted_residual_block(merge, 160, (3, 3), t=6, strides=2, n=3, alpha=alpha, block_id=14, name='merge')
    merge = _inverted_residual_block(merge, 320, (3, 3), t=6, strides=1, n=1, alpha=alpha, block_id=17, name='merge')
    merge = _conv_block(merge, 1280, alpha, (1, 1), strides=(1, 1), block_id=18, name='merge')

    merge = GlobalAveragePooling2D()(merge)
    merge = Dropout(0.4)(merge)
    merge = Dense(1024,activation='relu')(merge)
    merge = Dropout(0.4)(merge)

    out = Dense(classes,
                use_bias=False,
                kernel_regularizer=l2(weightdecay),
                activation='softmax',
                name='output')(merge)

    model = Model(inputs=inputs, outputs=out, name='mobilenetv2')
    print(model)
    # load weights
    return model

#############################################################################

def MobileNetV2_DANN(input_shape=None,
                    re_shape=(2,100,80,1),
                    DANN_strength = 0.1,
                    alpha=0.25,
                    depth_multiplier=1,
                    classifier_classes=3,
                    descriminator_classes=2,
                    batch_size = 32,
                    weightdecay=0.0002,
                    jitter=0.001,
                    input_tensor=None):

    """MobileNetv1
    This function defines a MobileNetv1 architectures.
    # Arguments
        inputs: Inuput Tensor, e.g. an image
        alpha: Width Multiplier
        depth_multiplier: Resolution Multiplier
        classes: number of labels
        weightdecay: weight decay for last layer
    # Returns
	five MobileNetv2 model stages."""
    if input_tensor is None:
        img_input = Input(shape=input_shape[0], name='input')
    inputs=img_input
    if jitter != 0:
        img_input_jitter = GaussianDropout(jitter)(img_input)
        shaped = Reshape(re_shape)(img_input_jitter)
    else:
        shaped = Reshape(re_shape)(img_input)
    def _lambda_unstack(x):
        import tensorflow as tf
        return tf.unstack(x,axis=1)
    shaped = Lambda(_lambda_unstack)(shaped)
    img_input1 = shaped[0]
    img_input2 = shaped[1]
    if(re_shape[0] == 4):
        img_input3 = shaped[2]
        img_input4 = shaped[3]
    if(re_shape[0] == 4):
        img_input = [img_input1, img_input2, img_input3, img_input4]
    else:
        img_input  = [img_input1, img_input2]

    branches = []
    names = ['x','y','px','py']
    for i in range(len(img_input)):
        branch = subnet(img_input[i], names[i], alpha)
        branches.append(branch)

    merge = concatenate(branches)

    merge = _inverted_residual_block(merge, 64,  (3, 3), t=6, strides=2, n=4, alpha=alpha, block_id=7, name='merge')
    merge = _inverted_residual_block(merge, 96,  (3, 3), t=6, strides=1, n=3, alpha=alpha, block_id=11, name='merge')
    merge = _inverted_residual_block(merge, 160, (3, 3), t=6, strides=2, n=3, alpha=alpha, block_id=14, name='merge')
    merge = _inverted_residual_block(merge, 320, (3, 3), t=6, strides=1, n=1, alpha=alpha, block_id=17, name='merge')
    merge = _conv_block(merge, 1280, alpha, (1, 1), strides=(1, 1), block_id=18, name='merge')

    av_pool = GlobalAveragePooling2D()(merge)

    merge = Dropout(0.4)(av_pool)
    merge = Dense(1024,activation='relu')(merge)
    merge = Dropout(0.4)(merge)

    #insert discriminator

    grl_layer = GradientReversal(1.0)
    feature_output_grl = grl_layer(av_pool)

    labeled_feature_output = Lambda(lambda x: K.switch(K.variable(1), K.concatenate([x[:int(batch_size//2)], x[:int(batch_size//2)]], axis=0), x), output_shape=lambda x: x[0:])(feature_output_grl)

    out = Dropout(0.5)(labeled_feature_output)
    out = Dense(128, activation="relu")(out)
    out = Dropout(0.5)(out)
    discriminator_output = Dense(descriminator_classes, activation="softmax", name="discriminator_output")(out)

    classifier_output = Dense(classifier_classes,
                              use_bias=False,
                              kernel_regularizer=l2(weightdecay),
                              activation='softmax',
                              name='output')(merge)


    model = Model(inputs=inputs, outputs=[classifier_output, discriminator_output], name='mobilenetv2')
    print(model)

    return model


### EXPERIMENTAL BLOCK ######################################################################

def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) *0.5* hp_lambda]

    g = get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


##############################################################################################

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), block_id=1, name=''):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        inputs: Input tensor of shape `(rows, cols, chans)`
            (with `channels_last` data format) or
            (chans, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
	Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name=name+'conv{}'.format(block_id))(inputs)
    x = BatchNormalization(axis=channel_axis, name=name+'conv{}_bn'.format(block_id))(x)
    return ReLU(6, name=name+'conv{}_relu'.format(block_id))(x)

def _bottleneck(inputs, filters, kernel, t, s, r=False, alpha=1.0, block_id=1, train_bn = False, name=''):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
	Output tensor.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    filters = int(alpha * filters)
    x = _conv_block(inputs, tchannel, alpha, (1, 1), (1, 1),block_id=block_id, name=name)
    x = DepthwiseConv2D(kernel,
                    strides=(s, s),
                    depth_multiplier=1,
                    padding='same',
                    name=name+'conv_dw_{}'.format(block_id))(x)
    x = BatchNormalization(axis=channel_axis,name=name+'conv_dw_{}_bn'.format(block_id))(x)
    x = ReLU(6, name=name+'conv_dw_{}_relu'.format(block_id))(x)
    x = Conv2D(filters,
                    (1, 1),
                    strides=(1, 1),
                    padding='same',
                    name=name+'conv_pw_{}'.format(block_id))(x)
    x = BatchNormalization(axis=channel_axis, name=name+'conv_pw_{}_bn'.format(block_id))(x, training=train_bn)
    if r:
        x = add([x, inputs], name=name+'res{}'.format(block_id))
    return x

def _inverted_residual_block(inputs, filters, kernel, t, strides, n, alpha, block_id, name=''):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
	Output tensor.
    """
    x = _bottleneck(inputs, filters, kernel, t, strides, False, alpha, block_id, name=name)
    for i in range(1, n):
        block_id += 1
        x = _bottleneck(x, filters, kernel, t, 1, True, alpha, block_id, name=name)
    return x

def subnet(x, name, alpha):
    x = _conv_block(x, 32, alpha, (3, 3), strides=(2, 2), block_id=0, name=name)
    x = _inverted_residual_block(x, 16,  (3, 3), t=1, strides=1, n=1, alpha=alpha, block_id=1, name=name)
    x = _inverted_residual_block(x, 24,  (3, 3), t=6, strides=2, n=2, alpha=alpha, block_id=2, name=name)
    x = _inverted_residual_block(x, 32,  (3, 3), t=6, strides=2, n=3, alpha=alpha, block_id=4, name=name)

    return x
