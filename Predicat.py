import os
import numpy as np
from keras.models import load_model
from keras.metrics import MeanIoU
from matplotlib import pyplot as plt
import segmentation_models_3D as sm
from data_gen import imageLoader  # Assuming your custom data generator
from tensorflow.keras.optimizers import Adam

# Data directories
val_img_dir = r"/home/d_rutvik/BraTS2020/input_data_2/val/images/"
val_mask_dir = r"/home/d_rutvik/BraTS2020/input_data_2/val/masks/"
val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

# Custom loss and metrics for loading the model
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# Loading the model with custom loss and metric
model_path = r'/home/d_rutvik/BraTS2020/src1.2/src1.1/best_model.keras'
my_model = load_model(model_path, custom_objects={
    'dice_loss_plus_1focal_loss': total_loss,
    'iou_score': sm.metrics.IOUScore(threshold=0.5)
})

# Recompile the model with a new optimizer
optimizer = Adam()  # Or any other optimizer you used previously
my_model.compile(optimizer=optimizer,
                 loss=total_loss,
                 metrics=[sm.metrics.IOUScore(threshold=0.5)])

# DataLoader setup
batch_size = 8
test_img_datagen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)

# Verify generator
test_image_batch, test_mask_batch = test_img_datagen.__next__()

# Calculate IoU for a batch
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# Prediction on a single test image
img_num = 82
test_img = np.load(r"/home/d_rutvik/BraTS2020/input_data_2/val/images/image_" + str(img_num) + ".npy")
test_mask = np.load(r"/home/d_rutvik/BraTS2020/input_data_2/val/masks/mask_" + str(img_num) + ".npy")
test_mask_argmax = np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

# Plot individual slices from test predictions for verification
n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_argmax[:, :, n_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:, :, n_slice])
plt.show()

# Optional: Continue training if necessary
history2 = my_model.fit(test_img_datagen,
                        steps_per_epoch=len(val_img_list) // batch_size,
                        epochs=1,
                        verbose=1,
                        validation_data=test_img_datagen,
                        validation_steps=len(val_img_list) // batch_size)
