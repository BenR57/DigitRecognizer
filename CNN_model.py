import numpy as np
#import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.layers.experimental.preprocessing import RandomRotation, RandomTranslation
from keras.callbacks import EarlyStopping


#TO TO : use convonlutional NN


class Load_model_for_pygame:
    
    def __init__(self, path):
        self.path = path
        self.model = keras.models.load_model(self.path)

    def __str__(self): 
        return "model based on " + self.path

    #Returns an array with probabilities    
    def predict_prob(self, input_image):
        #converting 28x28 input to a 1x28x28x1 array  
        x = np.expand_dims(input_image, axis=0)
        x = np.expand_dims(x, axis=3)
        return self.model.predict(x).reshape(10)
    
    #Returns the number with the highest probability
    def predict(self, input_image):
        #converting 28x28 input to a 1x28x28 array  
        x = np.expand_dims(input_image, axis=0)
        x = np.expand_dims(x, axis=3)
        return np.argmax(self.model.predict(x))


def create_and_train_model():    

    #Importing data
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #Normalizing the input
    x_train = keras.utils.normalize(x_train, axis = 1)
    x_test = keras.utils.normalize(x_test, axis = 1)
    #reshape the arrays for the CNN
    x_train_r = np.expand_dims(x_train, axis = 3)
    x_test_r = np.expand_dims(x_test, axis = 3)
    
    #Translating and Rotating the training set
    data_augmentation = Sequential([RandomTranslation(0.05, 0.05, fill_mode='constant'), 
                                    RandomRotation(0.02, fill_mode='constant')])
    x_train_r = data_augmentation(x_train_r) 
    
    
    model = Sequential()
    model.add(Conv2D(32,kernel_size=5,padding='same',activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(64,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    monitor = EarlyStopping(monitor = 'val_loss',
                            min_delta = 1e-3,
                            patience = 5,
                            verbose = 1,  
                            mode = 'auto',
                            restore_best_weights = True)
    
    model.fit(x_train_r, y_train, 
              validation_data = (x_test_r, y_test),
              callbacks = [monitor],
              verbose = 1,
              epochs = 20)

    model.save('number_recognition_CNN')


if __name__ == "__main__":
    create_and_train_model();





