# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 12:04:51 2016

@author: simon
"""
import matplotlib.pyplot as plt 
import os
def dxplot(obj,dir_name):
    '''arguement: 
        obj is a string.
        The list of obj:
        1.'b_losses' 'b_acces'
        2.'e_val_losses' 'e_val_acces'
        3. 'e_losses' 'e_acces'
    '''
    plt.figure()
    exec('b=history_b.'+obj)
    exec('s=history_s.'+obj)
    exec('r=history_r.'+obj)
    plt.plot(range(len(b)),b,'r',label='b')
    plt.plot(range(len(s)),s,'b',label='s')
    plt.plot(range(len(r)),r,'g',label='r')
    plt.legend()
    plt.title(obj)
    if not(os.path.exists('net2netpics/'+dir_name)):
        os.mkdir('net2netpics/'+dir_name)
    plt.savefig('net2netpics/'+dir_name+'/'+obj+'.jpg')
    
    
for obj in ['b_losses', 'b_acces','e_val_losses', 'e_val_acces','e_losses', 'e_acces']:
    dxplot(obj,'test')