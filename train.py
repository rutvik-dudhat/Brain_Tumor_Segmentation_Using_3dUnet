import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from u_net3d import build_unet_model
from data_gen import imageLoader
# Clear any previous sessions
K.clear_session()

# Set data directories
train_img_dir = "/home/d_rutvik/BraTS2020/input_data_2/train/images/"
train_mask_dir = "/home/d_rutvik/BraTS2020/input_data_2/train/masks/"
val_img_dir = "/home/d_rutvik/BraTS2020/input_data_2/val/images/"
val_mask_dir = "/home/d_rutvik/BraTS2020/input_data_2/val/masks/"

# Load model
input_shape = (128, 128, 128, 3)  # Adjust as needed
num_classes = 4
model = build_unet_model(input_shape=input_shape, num_classes=num_classes)

# Set training parameters
batch_size = 2  # Adjust based on GPU memory
epochs = 100

# Load image and mask lists
train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)
val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

# Load data
train_gen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
val_gen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)

# Calculate total steps
train_steps = len(train_img_list) // batch_size
val_steps = len(val_img_list) // batch_size

# Callbacks for saving the model at the end of every epoch
checkpoint = ModelCheckpoint('try_2_unet_model_epoch_{epoch:02d}.keras', save_best_only=False, monitor='val_loss', mode='min')
#early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=epochs,
                    steps_per_epoch=train_steps,
                    validation_steps=val_steps,
                    callbacks=[checkpoint])

# Save final model
model.save('try_2_unet_model_final.keras')

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
