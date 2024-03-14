# PRODIGY_ML_03
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix

# Define directories
train_dir = 'train'
test_dir = 'test'

# Define image dimensions and batch size
img_height, img_width = 150, 150
batch_size = 32

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for SVM classifier
x = Flatten()(base_model.output)
predictions = Dense(1, activation='sigmoid')(x)

# Create SVM model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=5,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f'Test accuracy: {test_acc}')

# Predictions
predictions = model.predict(test_generator)
y_pred = np.where(predictions > 0.5, 1, 0)

# Print classification report and confusion matrix
print(classification_report(test_generator.classes, y_pred))
print(confusion_matrix(test_generator.classes, y_pred))

