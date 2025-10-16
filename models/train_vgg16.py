# train_vgg16.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_loader import prepare_data

def create_vgg16():
    base_model = VGG16(weights=None, include_top=False, input_shape=(32, 32, 3))
    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    cifar10_dir = '/Users/Yoshua/cnn-research/cifar-10-batches-py'
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(cifar10_dir)
    
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    datagen.fit(x_train)

    vgg16 = create_vgg16()
    vgg16.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=50, validation_data=(x_val, y_val))
    vgg16.save_weights('vgg16_weights.h5')
