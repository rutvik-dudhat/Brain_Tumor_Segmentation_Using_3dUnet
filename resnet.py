import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from segmentation_models_3D import Unet
from datagennn import imageLoader  # Ensure this imports your custom imageLoader correctly
import tensorflow as tf
import matplotlib.pyplot as plt

# Paths
TRAIN_IMAGES_PATH = '/home/d_rutvik/BraTS2020/input_data_2/train/images/'
TRAIN_MASKS_PATH = '/home/d_rutvik/BraTS2020/input_data_2/train/masks/'
VAL_IMAGES_PATH = '/home/d_rutvik/BraTS2020/input_data_2/val/images/'
VAL_MASKS_PATH = '/home/d_rutvik/BraTS2020/input_data_2/val/masks/'
MODEL_SAVE_PATH = '/home/d_rutvik/BraTS2020/torch_try/neww_model_final123.keras'

# Hyperparameters
BATCH_SIZE = 2  # Adjust based on your GPU memory
EPOCHS = 100
LEARNING_RATE = 1e-4

# Load training and validation data
train_images = os.listdir(TRAIN_IMAGES_PATH)
train_masks = os.listdir(TRAIN_MASKS_PATH)

val_images = os.listdir(VAL_IMAGES_PATH)
val_masks = os.listdir(VAL_MASKS_PATH)

# Create data generators
train_datagen = imageLoader(
    img_dir=TRAIN_IMAGES_PATH,
    img_list=train_images,
    mask_dir=TRAIN_MASKS_PATH,
    mask_list=train_masks,
    batch_size=BATCH_SIZE
)

val_datagen = imageLoader(
    img_dir=VAL_IMAGES_PATH,
    img_list=val_images,
    mask_dir=VAL_MASKS_PATH,
    mask_list=val_masks,
    batch_size=BATCH_SIZE
)

# Build the pre-trained model
model = Unet(
    backbone_name='resnet34',
    input_shape=(128, 128, 128, 3),  # Adjust the input shape as needed
    classes=4,  # Number of classes
    activation='softmax',  # Activation function
    encoder_weights='imagenet'  # Using imagenet pre-trained weights
)

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for model saving
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1)


# Function to calculate mean IoU at the end of each epoch
def mean_iou(y_true, y_pred):
    # Convert predictions to class indices (argmax)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)

    # Calculate IoU for each class
    iou = tf.keras.metrics.MeanIoU(num_classes=4)
    iou.update_state(y_true, y_pred)
    return iou.result().numpy()


# Train the model with Mean IoU calculation at each epoch
history = model.fit(
    train_datagen,
    steps_per_epoch=len(train_images) // BATCH_SIZE,
    validation_data=val_datagen,
    validation_steps=len(val_images) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    verbose=1
)

# Calculate and print Mean IoU at the end of each epoch for both train and validation
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    # Train IoU
    train_pred = model.predict(train_datagen, steps=len(train_images) // BATCH_SIZE)
    train_iou = mean_iou(train_datagen.__getitem__(0)[1], train_pred)  # Update this line to use correct batch
    print(f"Train IoU: {train_iou}")

    # Validation IoU
    val_pred = model.predict(val_datagen, steps=len(val_images) // BATCH_SIZE)
    val_iou = mean_iou(val_datagen.__getitem__(0)[1], val_pred)  # Update this line to use correct batch
    print(f"Validation IoU: {val_iou}")

# Save the final model
model.save(MODEL_SAVE_PATH)

# Optional: Plotting training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
