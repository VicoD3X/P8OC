# ============================================================
# U-Net Mini — Modèle léger pour segmentation
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

# Import des métriques et pertes
from src.metrics import (
    dice_loss,
    iou_metric,
    dice_coef,
    pixel_accuracy
)


def unet_mini(input_shape=(256, 512, 3), num_classes=8):
    """
    Architecture U-Net Mini adaptée à la segmentation
    d’images Cityscapes (8 classes).
    """

    inputs = layers.Input(shape=input_shape)

    # ---------------------------
    # Encoder
    # ---------------------------
    x1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x1 = layers.Conv2D(32, 3, padding='same', activation='relu')(x1)
    p1 = layers.MaxPooling2D((2, 2))(x1)

    x2 = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    x2 = layers.Conv2D(64, 3, padding='same', activation='relu')(x2)
    p2 = layers.MaxPooling2D((2, 2))(x2)

    # ---------------------------
    # Bottleneck
    # ---------------------------
    b = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    b = layers.Conv2D(128, 3, padding='same', activation='relu')(b)

    # ---------------------------
    # Decoder
    # ---------------------------
    u3 = layers.UpSampling2D((2, 2))(b)
    u3 = layers.Concatenate()([u3, x2])
    x3 = layers.Conv2D(64, 3, padding='same', activation='relu')(u3)
    x3 = layers.Conv2D(64, 3, padding='same', activation='relu')(x3)

    u4 = layers.UpSampling2D((2, 2))(x3)
    u4 = layers.Concatenate()([u4, x1])
    x4 = layers.Conv2D(32, 3, padding='same', activation='relu')(u4)
    x4 = layers.Conv2D(32, 3, padding='same', activation='relu')(x4)

    # ---------------------------
    # Output
    # ---------------------------
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x4)

    model = models.Model(inputs=inputs, outputs=outputs, name="UNet_Mini")

    # ---------------------------
    # Compilation (Keras 3)
    # ---------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=dice_loss,
        metrics=[iou_metric, dice_coef, pixel_accuracy]
    )

    return model
