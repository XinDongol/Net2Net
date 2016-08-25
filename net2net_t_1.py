# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 21:04:36 2016

@author: C2i_User
"""


from keras.models import Sequential 
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(1, input_dim = 1, activation = 'sigmoid', name='l1'))
model.add(Dense(2 , activation = 'sigmoid', name='l2'))
model.add(Dense(1 , activation = 'sigmoid', name='l3'))

import numpy as np 
data = np.random.random((1000, 1))
labels = data>0.5


model.compile(
optimizer='sgd',
loss='mse',
metrics=['accuracy'])

result=model.fit(data, labels, nb_epoch=10, batch_size=32, verbose=2)

print 'Paras_l1=', model.get_layer('l1').get_weights()
print '\nParas_l2=', model.get_layer('l2').get_weights()
print '\nParas_l3=', model.get_layer('l3').get_weights()

value1 = model.get_layer('l1').get_weights()
value2 = np.mat([model.get_layer('l2').get_weights()[0][0],model.get_layer('l2').get_weights()[1]])
value2n = np.hstack((value2,value2[:,0]))
value2n = [np.array(value2n[0]),np.array(value2n[1])[0]]
value3 = model.get_layer('l3').get_weights()
value3[0][0]=0.5*value3[0][0]
value3[0] = np.vstack((value3[0],value3[0][0]))
   
model2 = Sequential()
#model2.add(Dense(1, init = l1_init, input_dim = 1, activation = 'sigmoid', name='N1'))
model2.add(Dense(1, input_dim = 1, activation = 'sigmoid', name='N1'))
model2.add(Dense(3, activation = 'sigmoid', name='N2'))
model2.add(Dense(1, activation = 'sigmoid', name='N3'))
model2.get_layer('N1').set_weights(value1)
model2.get_layer('N2').set_weights(value2n)
model2.get_layer('N3').set_weights(value3)

model2.compile(
optimizer='sgd',
loss='mse',
metrics=['accuracy'])

print 'Paras_N1b=', model2.get_layer('N1').get_weights()
print '\nParas_N2b=', model2.get_layer('N2').get_weights()
print '\nParas_N3b=', model2.get_layer('N3').get_weights()

print 'result_1:', model.predict(data[0:10],batch_size=1)
print '\nresult_2:', model2.predict(data[0:10],batch_size=1)

model.train_on_batch(data[0:32],labels[0:32])
model2.train_on_batch(data[0:32],labels[0:32])

print 'Paras_l1=', model.get_layer('l1').get_weights()
print '\nParas_l2=', model.get_layer('l2').get_weights()
print '\nParas_l3=', model.get_layer('l3').get_weights()

print '\n\nParas_N1=', model2.get_layer('N1').get_weights()
print '\nParas_N2=', model2.get_layer('N2').get_weights()
print '\nParas_N3=', model2.get_layer('N3').get_weights()
