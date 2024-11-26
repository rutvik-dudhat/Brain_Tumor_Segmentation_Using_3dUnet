import tensorflow as tf
from tensorflow.keras import layers, Model


def build_unet_model(input_shape=(128, 128, 128, 3), num_classes=4):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up1 = layers.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)
    up1 = layers.concatenate([up1, conv3], axis=-1)
    conv5 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up1)
    conv5 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)

    up2 = layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
    up2 = layers.concatenate([up2, conv2], axis=-1)
    conv6 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up2)
    conv6 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv6)

    up3 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
    up3 = layers.concatenate([up3, conv1], axis=-1)
    conv7 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up3)
    conv7 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv7)

    outputs = layers.Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv7)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
