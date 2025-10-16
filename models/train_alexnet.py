import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from data_loader import prepare_data

def create_alexnet():
    model = Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(96, (11, 11), strides=4, activation='relu', padding='same'),
        MaxPooling2D((3, 3), strides=2, padding='same'),
        Conv2D(256, (5, 5), padding='same', activation='relu'),
        MaxPooling2D((3, 3), strides=2, padding='same'),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((3, 3), strides=2, padding='same'),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    cifar10_dir = '/Users/Yoshua/cnn-research/cifar-10-batches-py'
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(cifar10_dir)
    
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    datagen.fit(x_train)

    alexnet = create_alexnet()
    
    # Define the checkpoint callback
    checkpoint = ModelCheckpoint('alexnet_weights.keras', save_best_only=True, monitor='val_loss', mode='min')

    # Train the model with checkpointing
    alexnet.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=50, validation_data=(x_val, y_val), callbacks=[checkpoint])
