import keras
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar100
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = 100
no_epochs = 100
optimizer = Adam()
validation_split = 0.2
verbosity = 1
L2_DECAY_RATE = 0.0005
INIT_DROPOUT_RATE = 0.5


(input_train, target_train), (input_test, target_test) = cifar100.load_data(label_mode='fine')
input_shape = (img_width, img_height, img_num_channels)
# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# # Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# model = Sequential()
# # model.add(ZeroPadding2D(4, input_shape=input_train.shape[1:]))
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# # Stack 1:
# model.add(Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(0.01)))
# # model.add(Activation('elu'))
# model.add(Dense(256, activation='elu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # model.add(Dropout(INIT_DROPOUT_RATE))
# # Stack 2:
# model.add(Conv2D(384, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# model.add(Conv2D(384, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# model.add(Conv2D(640, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# model.add(Conv2D(640, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# # model.add(Activation('elu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # model.add(Dropout(INIT_DROPOUT_RATE))
# # Stack 3:
# model.add(Conv2D(640, (3, 3), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# model.add(Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# model.add(Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# model.add(Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# # model.add(Activation('elu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # model.add(Dropout(INIT_DROPOUT_RATE))
# # Stack 4:
# model.add(Conv2D(768, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# model.add(Conv2D(896, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# model.add(Conv2D(896, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# # model.add(Activation('elu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # model.add(Dropout(INIT_DROPOUT_RATE))
# # Stack 5:
# model.add(Conv2D(896, (3, 3), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# model.add(Conv2D(1024, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# model.add(Conv2D(1024, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# # model.add(Activation('elu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # model.add(Dropout(INIT_DROPOUT_RATE))
# # Stack 6:
# model.add(Conv2D(1024, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# model.add(Conv2D(1152, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# # model.add(Activation('elu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # model.add(Dropout(INIT_DROPOUT_RATE))
# # Stack 7:
# model.add(Conv2D(1152, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
# # model.add(Activation('elu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # model.add(Dropout(INIT_DROPOUT_RATE))
# model.add(Flatten())
# model.add(Dense(no_classes, activation='softmax'))

callbacks_model = [EarlyStopping('val_loss', patience=20),
             ModelCheckpoint('cifar100Model2_keras.h5', save_best_only=True)]
# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

# Fit data to model
history = model.fit(input_train, target_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split,
					callbacks=callbacks_model,)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Visualize history
# Plot history: Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()

# print images from dataset
# print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
# print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# # plot first few images
# for i in range(9):
# 	# define subplot
# 	pyplot.subplot(330 + 1 + i)
# 	# plot raw pixel data
# 	pyplot.title(trainy[i])
# 	pyplot.imshow(trainX[i])
# # show the figure
# pyplot.show()