# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
import scipy.io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
# Dataset Loading

a = scipy.io.loadmat('Dataset_for_CNN2')

X_train = a['X_train']
X_test = a['X_test']
y_train = a['y_train'].astype(int)
y_test = a['y_test']



X_train = X_train.reshape(X_train.shape[0],128, 128,1)
X_test = X_test.reshape(X_test.shape[0],128, 128,1)
X_train
# Preprocess Input Data
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 2).astype(int)
Y_test = np_utils.to_categorical(y_test, 2).astype(int)


 #Preprocess Class Labels
#converting to one hot notation


 
# 7. Define model architecture
model = Sequential()
model.add(Convolution2D(32, 8, 8, activation='relu', input_shape=(128,128,1)))
model.add(Convolution2D(64, 4, 4, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Convolution2D(64, 4, 4, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
 
model.summary()
# 8. Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
val=model.fit(X_train, Y_train, 
          batch_size=32, epochs=100, verbose=1)

 
# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
model.save('D:/Breast Cancer/asif/Code/trained_models/model.h5')
print(score)