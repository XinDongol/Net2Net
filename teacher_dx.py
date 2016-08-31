# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:52:58 2016

@author: simon
"""

def make_teacher_model_dx(train_data, validation_data, nb_epoch=3, wei_from='new', do_save='0'):
    '''Train a simple CNN as teacher model.
    '''
    if wei_from == 'new':
        model = Sequential()
        model.add(Conv2D(64, 3, 3, input_shape=input_shape,
                         border_mode='same', name='conv1'))
        model.add(MaxPooling2D(name='pool1'))
        model.add(Conv2D(64, 3, 3, border_mode='same', name='conv2'))
        model.add(MaxPooling2D(name='pool2'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(32, activation='relu', name='fc1'))
        model.add(Dense(nb_class, activation='softmax', name='fc2'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.01, momentum=0.9),
                      metrics=['accuracy'])
        
        train_x, train_y = train_data
        history = LossHistory()
        model.fit(train_x, train_y, nb_epoch=nb_epoch,
                            validation_data=validation_data, batch_size = 256, callbacks=[history])
                            
        if do_save!='0':
            model.save(do_save)
                        
    else:
        from keras.models import load_model
        model = load_model(wei_from)
        history=[]
    return model, history