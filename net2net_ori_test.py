# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:14:56 2016

@author: C2i_User
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist

input_shape = (1, 28, 28)  # image shape
nb_class = 10  # number of class


# load and pre-process data
def preprocess_input(x):
    return x.reshape((-1, ) + input_shape) / 255.


def preprocess_output(y):
    return np_utils.to_categorical(y)

(train_x, train_y), (validation_x, validation_y) = mnist.load_data()
train_x, validation_x = map(preprocess_input, [train_x, validation_x])
train_y, validation_y = map(preprocess_output, [train_y, validation_y])
print('Loading MNIST data...')
print('train_x shape:', train_x.shape, 'train_y shape:', train_y.shape)
print('validation_x shape:', validation_x.shape,
      'validation_y shape', validation_y.shape)


# knowledge transfer algorithms
def wider2net_conv2d(teacher_w1, teacher_b1, teacher_w2, new_width, init):
    '''Get initial weights for a wider conv2d layer with a bigger nb_filter,
    by 'random-padding' or 'net2wider'.
    # Arguments
        teacher_w1: `weight` of conv2d layer to become wider,
          of shape (nb_filter1, nb_channel1, kh1, kw1)
        teacher_b1: `bias` of conv2d layer to become wider,
          of shape (nb_filter1, )
        teacher_w2: `weight` of next connected conv2d layer,
          of shape (nb_filter2, nb_channel2, kh2, kw2)
        new_width: new `nb_filter` for the wider conv2d layer
        init: initialization algorithm for new weights,
          either 'random-pad' or 'net2wider'
    '''
    assert teacher_w1.shape[0] == teacher_w2.shape[1], (
        'successive layers from teacher model should have compatible shapes')
    assert teacher_w1.shape[0] == teacher_b1.shape[0], (
        'weight and bias from same layer should have compatible shapes')
    assert new_width > teacher_w1.shape[0], (
        'new width (nb_filter) should be bigger than the existing one')

    n = new_width - teacher_w1.shape[0]
    if init == 'random-pad':
        new_w1 = np.random.normal(0, 0.1, size=(n, ) + teacher_w1.shape[1:])
        new_b1 = np.ones(n) * 0.1
        new_w2 = np.random.normal(0, 0.1, size=(
            teacher_w2.shape[0], n) + teacher_w2.shape[2:])
    elif init == 'net2wider':
        index = np.random.randint(teacher_w1.shape[0], size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[index, :, :, :]
        new_b1 = teacher_b1[index]
        new_w2 = teacher_w2[:, index, :, :] / factors.reshape((1, -1, 1, 1))
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)

    student_w1 = np.concatenate((teacher_w1, new_w1), axis=0)
    if init == 'random-pad':
        student_w2 = np.concatenate((teacher_w2, new_w2), axis=1)
    elif init == 'net2wider':
        # add small noise to break symmetry, so that student model will have
        # full capacity later
        noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
        student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=1)
        student_w2[:, index, :, :] = new_w2
    student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

    return student_w1, student_b1, student_w2


def wider2net_fc(teacher_w1, teacher_b1, teacher_w2, new_width, init):
    '''Get initial weights for a wider fully connected (dense) layer
       with a bigger nout, by 'random-padding' or 'net2wider'.
    # Arguments
        teacher_w1: `weight` of fc layer to become wider,
          of shape (nin1, nout1)
        teacher_b1: `bias` of fc layer to become wider,
          of shape (nout1, )
        teacher_w2: `weight` of next connected fc layer,
          of shape (nin2, nout2)
        new_width: new `nout` for the wider fc layer
        init: initialization algorithm for new weights,
          either 'random-pad' or 'net2wider'
    '''
    assert teacher_w1.shape[1] == teacher_w2.shape[0], (
        'successive layers from teacher model should have compatible shapes')
    assert teacher_w1.shape[1] == teacher_b1.shape[0], (
        'weight and bias from same layer should have compatible shapes')
    assert new_width > teacher_w1.shape[1], (
        'new width (nout) should be bigger than the existing one')

    n = new_width - teacher_w1.shape[1]
    if init == 'random-pad':
        new_w1 = np.random.normal(0, 0.1, size=(teacher_w1.shape[0], n))
        new_b1 = np.ones(n) * 0.1
        new_w2 = np.random.normal(0, 0.1, size=(n, teacher_w2.shape[1]))
    elif init == 'net2wider':
        index = np.random.randint(teacher_w1.shape[1], size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[:, index]
        new_b1 = teacher_b1[index]
        new_w2 = teacher_w2[index, :] / factors[:, np.newaxis]
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)

    student_w1 = np.concatenate((teacher_w1, new_w1), axis=1)
    if init == 'random-pad':
        student_w2 = np.concatenate((teacher_w2, new_w2), axis=0)
    elif init == 'net2wider':
        # add small noise to break symmetry, so that student model will have
        # full capacity later
        noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
        student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=0)
        student_w2[index, :] = new_w2
    student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

    return student_w1, student_b1, student_w2


def deeper2net_conv2d(teacher_w):
    '''Get initial weights for a deeper conv2d layer by net2deeper'.
    # Arguments
        teacher_w: `weight` of previous conv2d layer,
          of shape (nb_filter, nb_channel, kh, kw)
    '''
    nb_filter, nb_channel, kh, kw = teacher_w.shape
    student_w = np.zeros((nb_filter, nb_filter, kh, kw))
    for i in xrange(nb_filter):
        student_w[i, i, (kh - 1) / 2, (kw - 1) / 2] = 1.
    student_b = np.zeros(nb_filter)
    return student_w, student_b


def copy_weights(teacher_model, student_model, layer_names):
    '''Copy weights from teacher_model to student_model,
     for layers with names listed in layer_names
    '''
    for name in layer_names:
        weights = teacher_model.get_layer(name=name).get_weights()
        student_model.get_layer(name=name).set_weights(weights)


# methods to construct teacher_model and student_models
def make_teacher_model(train_data, validation_data, nb_epoch=3):
    '''Train a simple CNN as teacher model.
    '''
    model = Sequential()
    model.add(Conv2D(64, 3, 3, input_shape=input_shape,
                     border_mode='same', name='conv1'))
    model.add(MaxPooling2D(name='pool1'))
    model.add(Conv2D(64, 3, 3, border_mode='same', name='conv2'))
    model.add(MaxPooling2D(name='pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, activation='relu', name='fc1'))
    model.add(Dense(nb_class, activation='softmax', name='fc2'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, momentum=0.9),
                  metrics=['accuracy'])

    train_x, train_y = train_data
    history = model.fit(train_x, train_y, nb_epoch=nb_epoch,
                        validation_data=validation_data)
    return model, history


def make_wider_student_model(teacher_model, train_data,
                             validation_data, init, nb_epoch=3):
    '''Train a wider student model based on teacher_model,
       with either 'random-pad' (baseline) or 'net2wider'
    '''
    new_conv1_width = 128
    new_fc1_width = 128

    model = Sequential()
    # a wider conv1 compared to teacher_model
    model.add(Conv2D(new_conv1_width, 3, 3, input_shape=input_shape,
                     border_mode='same', name='conv1'))
    model.add(MaxPooling2D(name='pool1'))
    model.add(Conv2D(64, 3, 3, border_mode='same', name='conv2'))
    model.add(MaxPooling2D(name='pool2'))
    model.add(Flatten(name='flatten'))
    # a wider fc1 compared to teacher model
    model.add(Dense(new_fc1_width, activation='relu', name='fc1'))
    model.add(Dense(nb_class, activation='softmax', name='fc2'))

    # The weights for other layers need to be copied from teacher_model
    # to student_model, except for widened layers
    # and their immediate downstreams, which will be initialized separately.
    # For this example there are no other layers that need to be copied.

    w_conv1, b_conv1 = teacher_model.get_layer('conv1').get_weights()
    w_conv2, b_conv2 = teacher_model.get_layer('conv2').get_weights()
    new_w_conv1, new_b_conv1, new_w_conv2 = wider2net_conv2d(
        w_conv1, b_conv1, w_conv2, new_conv1_width, init)
    model.get_layer('conv1').set_weights([new_w_conv1, new_b_conv1])
    model.get_layer('conv2').set_weights([new_w_conv2, b_conv2])

    w_fc1, b_fc1 = teacher_model.get_layer('fc1').get_weights()
    w_fc2, b_fc2 = teacher_model.get_layer('fc2').get_weights()
    new_w_fc1, new_b_fc1, new_w_fc2 = wider2net_fc(
        w_fc1, b_fc1, w_fc2, new_fc1_width, init)
    model.get_layer('fc1').set_weights([new_w_fc1, new_b_fc1])
    model.get_layer('fc2').set_weights([new_w_fc2, b_fc2])

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, momentum=0.9),
                  metrics=['accuracy'])

    train_x, train_y = train_data
    history = model.fit(train_x, train_y, nb_epoch=nb_epoch,
                        validation_data=validation_data)
    return model, history


def make_deeper_student_model(teacher_model, train_data,
                              validation_data, init, nb_epoch=3):
    '''Train a deeper student model based on teacher_model,
       with either 'random-init' (baseline) or 'net2deeper'
    '''
    model = Sequential()
    model.add(Conv2D(64, 3, 3, input_shape=input_shape,
                     border_mode='same', name='conv1'))
    model.add(MaxPooling2D(name='pool1'))
    model.add(Conv2D(64, 3, 3, border_mode='same', name='conv2'))
    # add another conv2d layer to make original conv2 deeper
    if init == 'net2deeper':
        prev_w, _ = model.get_layer('conv2').get_weights()
        new_weights = deeper2net_conv2d(prev_w)
        model.add(Conv2D(64, 3, 3, border_mode='same',
                         name='conv2-deeper', weights=new_weights))
    elif init == 'random-init':
        model.add(Conv2D(64, 3, 3, border_mode='same', name='conv2-deeper'))
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)
    model.add(MaxPooling2D(name='pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, activation='relu', name='fc1'))
    # add another fc layer to make original fc1 deeper
    if init == 'net2deeper':
        # net2deeper for fc layer with relu, is just an identity initializer
        model.add(Dense(64, init='identity',
                        activation='relu', name='fc1-deeper'))
    elif init == 'random-init':
        model.add(Dense(64, activation='relu', name='fc1-deeper'))
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)
    model.add(Dense(nb_class, activation='softmax', name='fc2'))

    # copy weights for other layers
    copy_weights(teacher_model, model, layer_names=[
                 'conv1', 'conv2', 'fc1', 'fc2'])

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, momentum=0.9),
                  metrics=['accuracy'])

    train_x, train_y = train_data
    history = model.fit(train_x, train_y, nb_epoch=nb_epoch,
                        validation_data=validation_data)
    return model, history


# experiments setup
def net2wider_experiment():
    '''Benchmark performances of
    (1) a teacher model,
    (2) a wider student model with `random_pad` initializer
    (3) a wider student model with `Net2WiderNet` initializer
    '''
    train_data = (train_x, train_y)
    validation_data = (validation_x, validation_y)
    print('\nExperiment of Net2WiderNet ...')
    print('\nbuilding teacher model ...')
    teacher_model, _ = make_teacher_model(train_data,
                                          validation_data,
                                          nb_epoch=3)

    print('\nbuilding wider student model by random padding ...')
    make_wider_student_model(teacher_model, train_data,
                             validation_data, 'random-pad',
                             nb_epoch=3)
    print('\nbuilding wider student model by net2wider ...')
    make_wider_student_model(teacher_model, train_data,
                             validation_data, 'net2wider',
                             nb_epoch=3)


def net2deeper_experiment():
    '''Benchmark performances of
    (1) a teacher model,
    (2) a deeper student model with `random_init` initializer
    (3) a deeper student model with `Net2DeeperNet` initializer
    '''
    train_data = (train_x, train_y)
    validation_data = (validation_x, validation_y)
    print('\nExperiment of Net2DeeperNet ...')
    print('\nbuilding teacher model ...')
    teacher_model, _ = make_teacher_model(train_data,
                                          validation_data,
                                          nb_epoch=3)

    print('\nbuilding deeper student model by random init ...')
    make_deeper_student_model(teacher_model, train_data,
                              validation_data, 'random-init',
                              nb_epoch=3)
    print('\nbuilding deeper student model by net2deeper ...')
    make_deeper_student_model(teacher_model, train_data,
                              validation_data, 'net2deeper',
                              nb_epoch=3)

# run the experiments

train_data = (train_x[0:1000], train_y[0:1000])
validation_data = (validation_x, validation_y)
print('\nonly building teacher model ...')
teacher_model, teacher_history = make_teacher_model(train_data,\
                                          validation_data,\
                                          nb_epoch=1)
                                         
                                                                                

