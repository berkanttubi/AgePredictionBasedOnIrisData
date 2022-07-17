'''
Berkant Tuğberk Demirtaş – 2315232
Sabahattin Yiğit Günaştı - 2315281

'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import GlorotUniform, RandomUniform
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy.random import seed



def read_data(path):
    number_of_data=0
    geo_or_txt=0
    df = pd.read_csv(path, sep='\n')
    data=[]
    nump = np.array(df)
    for x in nump:
        if x[0][0]!='@':
            data.append(x[0].split(','))
            number_of_data=number_of_data+1
        else:
            geo_or_txt=geo_or_txt+1

    labels = [int(data[i][-1])-1 for i in range(number_of_data)]
    raw_data=[data[i][:-1] for i in range(number_of_data)]
    labels=np.array(labels)
    raw_data = np.array(raw_data)
    if geo_or_txt>15:
        raw_data=raw_data.astype('int')   
    else:
        raw_data=raw_data.astype('float')  


    return raw_data,labels


def texture_NN(X, y, X_test, y_test,takennum):
    '''
    X = Texture Data Train
    y = Texture Data Train Label
    
    X_test = Texture Data Test
    y_test = Texture Data Test Label

    takennum = Number of Hidden Layer
    '''

    # Arrange seed for repeatable results
    seed(1)
    tf.random.set_seed(1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=True, stratify=y, random_state=1)
    
           
    model = Sequential()
    model.add(Flatten())       
        
        
    a=[128,64,48] # Neuron numbers for giving layer

    # If there is hidden layer, add it dynamically
    if takennum !=0:
        for i in range(takennum):
                    model.add(Dense(a[i], activation='relu', kernel_initializer=RandomUniform(seed=1)))
                    if i==0:
                        model.add(Dropout(0.8))
                    else:
                        model.add(Dropout(0.6))
            
    model.add(Dense(3, activation='softmax'))
    
    # compile the model
    opt = RMSprop(learning_rate=0.0001, rho=0.9, momentum=0.0)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['acc'])
    
    # Early Stopping Method for choosing epoch number dynamically
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
    
    batches = 32
    epochs=200
    

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batches, verbose=1 ,callbacks=early_stop)
    
    # evaluate the model
    _, validation_accuracy = model.evaluate(X_val, y_val, batch_size=batches)
    _, test_accuracy = model.evaluate(X_test, y_test, batch_size=batches)

    return validation_accuracy,test_accuracy
    
    
    
    
def geo_NN(X, y, X_test, y_test,takennum):

    '''
    X = Texture Data Train
    y = Texture Data Train Label
    
    X_test = Texture Data Test
    y_test = Texture Data Test Label

    takennum = Number of Hidden Layer    
    '''

    # Arrange seed for repeatable results
    seed(1)
    tf.random.set_seed(1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=True, stratify=y, random_state=1)


        
    model = Sequential()
    model.add(Flatten())       



    a=[128,64,48]
    
    if takennum !=0:
        for i in range(takennum):
                    model.add(Dense(a[i], activation='relu'))
                    if i==0:
                        model.add(Dropout(0.8))
                    else:
                        model.add(Dropout(0.6))
                
    
    #model.add(Dropout(0.2))

    model.add(Dense(3, activation='softmax'))

    # compile the model
    opt = RMSprop(learning_rate=0.000001, rho=0.9, momentum=0.0)
    #model.compile(optimizer=opt, loss='binary_crossentropy')
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['acc'])
    
    # Early Stopping Method for choosing epoch number dynamically
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
    
    batches = 8
    epochs=500


    # fit model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batches, verbose=1,callbacks=early_stop )

    # evaluate the model

    _, validation_accuracy = model.evaluate(X_val, y_val, batch_size=batches)
    _, test_accuracy = model.evaluate(X_test, y_test, batch_size=batches)

    return validation_accuracy,test_accuracy


    
    

def combine_NN(X, y, X_test, y_test,takennum):
    '''
    X = Texture Data Train
    y = Texture Data Train Label
    
    X_test = Texture Data Test
    y_test = Texture Data Test Label

    takennum = Number of Hidden Layer    
    '''

    # Arrange seed for repeatable results
    seed(1)
    tf.random.set_seed(1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=True, stratify=y, random_state=1)

    # 0 - Hidden Layer NN #
    
        
        
    model = Sequential()
    model.add(Flatten())  
    
        
    a=[128,64,48]
    
    if takennum !=0:
        for i in range(takennum):
                    model.add(Dense(a[i], activation='relu', kernel_initializer=RandomUniform(seed=1)))
                    if i==0:
                        model.add(Dropout(0.8))
                    else:
                        model.add(Dropout(0.6))  
        

    model.add(Dense(3, activation='softmax'))

    # compile the model
    opt = RMSprop(learning_rate=0.00001, rho=0.9, momentum=0.0)
    #model.compile(optimizer=opt, loss='binary_crossentropy')
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['acc'])
    
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
    
    batches = 64
    epochs=200

    # fit model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batches, verbose=1,callbacks=early_stop )


    _, validation_accuracy = model.evaluate(X_val, y_val, batch_size=batches)
    _, test_accuracy = model.evaluate(X_test, y_test, batch_size=batches)

    return validation_accuracy,test_accuracy
    

geometic_test_path = "IrisGeometicFeatures_TestingSet.txt"
geometic_train_path = "IrisGeometicFeatures_TrainingSet.txt"

texture_test_path = "IrisTextureFeatures_TestingSet.txt"
texture_train_path = "IrisTextureFeatures_TrainingSet.txt"

geo_x_train, geo_y_train = read_data(geometic_train_path)
geo_x_test, geo_y_test = read_data(geometic_test_path)
txt_x_train, txt_y_train = read_data(texture_train_path)
txt_x_test, txt_y_test = read_data(texture_test_path)

all_x_train = np.concatenate((txt_x_train, geo_x_train), axis=1)
all_x_test = np.concatenate((txt_x_test, geo_x_test), axis=1)


while(True):
    numberOfN= int(input("Enter the number of hidden layer: "))
    choice = int(input("Enter the feature choice: texture(1), geometic(2), both(3), exit(4): "))
    if choice==1:
        val_acc,test_acc=texture_NN(txt_x_train, txt_y_train,txt_x_test, txt_y_test,numberOfN)
    elif choice==2:
        val_acc,test_acc=geo_NN(geo_x_train, geo_y_train,geo_x_test, geo_y_test,numberOfN)
    elif choice==3:
        val_acc,test_acc=combine_NN(all_x_train, txt_y_train, all_x_test, txt_y_test, numberOfN)
    else:
        break
    
    print('Validation Accuracy: %f, Test Accuracy: %.3f'  % (val_acc*100, test_acc*100))
