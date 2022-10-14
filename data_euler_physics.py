# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:33:05 2022

@author: General
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def odeEuler(f,a0,delta_s,t):
    
    y = np.zeros((len(delta_s),60))
    
    for i in range(0,60):
        y[0,i] = a0
        
        for n in range(0,len(delta_s)-1):
            y[n+1,i] = y[n,i] + f(y[n,i],delta_s[n,i])*(t[n+1] - t[n])
    k=y
    return k
        # return y

def f(y,delta_s):
    [C, m] = [1.5E-11, 3.8]
    F=1
    f=C*(F*delta_s*np.sqrt(np.pi*y))**m
    return f

def Normalization(S_low, S_up, a_low, a_up,input1,input2):
   
        low_bound_S   = S_low
        upper_bound_S = S_up
        low_bound_a   = a_low
        upper_bound_a = a_up
    
        n_delta_s  = (input1 -low_bound_S) / (upper_bound_S - low_bound_S)
        n_delta_a  = (input2 - low_bound_a) / (upper_bound_a - low_bound_a)
        return n_delta_s,n_delta_a

    
a0=0.005
t = np.linspace(0,1,7300)
delta_s = np.asarray(pd.read_csv('./data/strain.csv'))

delta_s=np.transpose(delta_s)


# train_xx=delta_s.reshape(60,7300,1)


a     = np.asarray(pd.read_csv('./data/a0.csv'))[0,0]*np.ones((delta_s.shape[0],1))
y = odeEuler(f,a0,delta_s,t)
a_train=np.asarray(pd.read_csv('./data/atrain.csv'))

n_delta_s,n_delta_a=Normalization(np.min(delta_s), np.max(delta_s), np.min(y), np.max(y),delta_s,y)
train_xx=np.zeros((60, 2, 7300))
train_x=n_delta_s.reshape(60,1,7300)
train_y=a_train.reshape(60,1,1)
crack_len=n_delta_a.reshape(60,7300,1)


for i in range(0,60):
    train_xx[i,0,:]=train_x[i,0,:]
    train_xx[i,1,:]=crack_len[i,0,:]
    
x_test=np.asarray(pd.read_csv('./data/Stest.csv'))

a_test=np.asarray(pd.read_csv('./data/atest.csv'))
# y_test=np.asarray(pd.read_csv('./data/aatest.csv'))
x_test=x_test.reshape(300,1,7300)
aa_test=a_test.reshape(300,1,7300)
xx_test=np.zeros((300, 2, 7300))
for i in range(0,300):
    xx_test[i,0,:]=x_test[i,0,:]
    xx_test[i,1,:]=aa_test[i,0,:]
    
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
 
# trainx=np.reshape(delta_s, (1,delta_s.shape[0],delta_s.shape[1]))
# output=np.reshape(y, (1,y.shape[0],y.shape[1]))
model=Sequential()
# model.add(SimpleRNN(128, input_shape=(2,7300),return_sequences=False))
#model.add(GRU(32, input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences=True))
model.add(LSTM(128, input_shape=(2,7300),return_sequences=False))
model.add(Dense(15, activation='tanh'))
model.add(Dense(15, activation='tanh'))
model.add(Dense(1))
# model.add(TimeDistributed(Dense(1)))  #there is no difference between this and model.add(Dense(1))...
# does not make sense to use metrics=['acc'], see https://stackoverflow.com/questions/41819457/zero-accuracy-training-a-neural-network-in-keras
model.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mse'])

 
# after every epoch, we save the model, this is the absolute path on my C: drive, so the path is
# C:\python_files\system_identification\models\
# filepath="C:\\Nikhil Mahar\\sem2\pinn\\codes\\pinn_ode_tutorial-master\\first_order_ode\\weights-{epoch:02d}-{val_loss:.6f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# callbacks_list = [checkpoint]    
# history=model.fit(train_xx,train_y , epochs=10, callbacks=callbacks_list, validation_data=(xx_test, y_test),batch_size=1,verbose=1)
 # history=model.fit(train_xx,train_y , epochs=5,batch_size=1,verbose=1)
mckp = ModelCheckpoint(filepath = "./savedmodels/cp.ckpt", monitor = 'loss', verbose = 1,
                        save_best_only = True, mode = 'min', save_weights_only = True)
# aPred_before = model.predict_on_batch(train_xx)[:,:]
history = model.fit(train_xx, train_y, epochs=500, steps_per_epoch=1, verbose=1, callbacks=[mckp])
# aPred = model.predict_on_batch(train_xx)[:,:]

aBefore = model.predict_on_batch(xx_test)
# use the test data to predict the model response
# testPredict = model.predict(xx_test) 
# loading weights from trained model
model.load_weights("./savedmodels/cp.ckpt")
aAfter = model.predict(xx_test)
# loss=history.history['loss']
# epochs=range(1,len(loss)+1)
# plt.figure()
# plt.plot(epochs, loss,'b', label='Training loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.xscale('log')
# #plt.yscale('log')
# plt.legend()

# plotting predictions
fig = plt.figure()
plt.plot(np.array(history.history['loss']))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(which='both')
plt.show()

# fig = plt.figure()
# plt.plot([0,0.05],[0,0.05],'--k')
# plt.plot(a_train, aPred_before, 'o', label = 'before training')
# plt.plot(a_train, aPred, 's', label = 'after training')
# plt.xlabel("actual crack length (m)")
# plt.ylabel("predicted crack length (m)")
# plt.legend(loc = 'upper center',facecolor = 'w')
# plt.grid(which='both')
# plt.show

fig = plt.figure()
plt.plot([0,0.05],[0,0.05],'--k')
plt.plot(a_test[:,-1],aBefore[:,:],'o', label = 'before training')
plt.plot(a_test[:,-1],aAfter[:,:], 's', label = 'after training')
plt.xlabel("actual crack length (m)")
plt.ylabel("predicted crack length (m)")
plt.legend(loc = 'upper center',facecolor = 'w')
plt.grid(which='both')
plt.show()

 # plotting predictions
fig = plt.figure()
# plt.plot(aBefore[:,-1],'k', label = 'before trainingl')
plt.plot(a_test[:,-1],'o', label = 'actual')
plt.plot(aAfter[:,-1], 's', label = 'after training')
plt.xlabel("machine")
plt.ylabel("crack length (m)")
plt.legend(loc = 'upper center',facecolor = 'w')
plt.grid(which='both')
plt.show()

# =np.mse(a_test[:,-1],aAfter[:,-1])
test_mse=np.square(np.subtract(a_test[:,-1],aAfter[:,-1])).mean()
print("test loss",test_mse)