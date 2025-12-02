# ============================================================
# U-Net basé sur MobileNetV2 — Architecture légère optimisée
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models     # type: ignore
from tensorflow.keras.applications import MobileNetV2   # type: ignore

# Import des métriques, pertes et outils
from src.metrics import (
    dice_loss,
    iou_metric,
    dice_coef,
    pixel_accuracy
)


def unet_mobilenetv2(input_shape=(256, 512, 3), num_classes=8):
    """
    Architecture U-Net utilisant MobileNetV2 comme encodeur.
    Version compacte adaptée à une utilisation rapide / embarquée.
    """

    # ---------------------------
    # 1) ENCODER : MobileNetV2
    # ---------------------------
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Gel des poids du backbone pour stabilisation / CPU-friendly
    base.trainable = False

    # Extraction de trois niveaux de features (mobilenet blocks)
    c1 = base.get_layer("block_3_expand_relu").output   # ~64x128x96
    c2 = base.get_layer("block_6_expand_relu").output   # ~32x64x192
    c3 = base.get_layer("block_13_expand_relu").output  # ~16x32x576

    # ---------------------------
    # 2) BOTTLENECK
    # ---------------------------
    b = layers.Conv2D(512, 3, padding="same", activation="relu")(c3)
    b = layers.Conv2D(512, 3, padding="same", activation="relu")(b)

    # ---------------------------
    # 3) DECODER
    # ---------------------------

    # Up 1 : 16x32 -> 32x64
    u1 = layers.UpSampling2D((2, 2))(b)
    u1 = layers.Concatenate()([u1, c2])
    x1 = layers.Conv2D(256, 3, padding="same", activation="relu")(u1)
    x1 = layers.Conv2D(256, 3, padding="same", activation="relu")(x1)

    # Up 2 : 32x64 -> 64x128
    u2 = layers.UpSampling2D((2, 2))(x1)
    u2 = layers.Concatenate()([u2, c1])
    x2 = layers.Conv2D(128, 3, padding="same", activation="relu")(u2)
    x2 = layers.Conv2D(128, 3, padding="same", activation="relu")(x2)

    # Up 3 : 64x128 -> 128x256
    u3 = layers.UpSampling2D((2, 2))(x2)
    x3 = layers.Conv2D(64, 3, padding="same", activation="relu")(u3)
    x3 = layers.Conv2D(64, 3, padding="same", activation="relu")(x3)

    # Up 4 : 128x256 -> 256x512
    u4 = layers.UpSampling2D((2, 2))(x3)
    x4 = layers.Conv2D(32, 3, padding="same", activation="relu")(u4)
    x4 = layers.Conv2D(32, 3, padding="same", activation="relu")(x4)

    # ---------------------------
    # 4) OUTPUT
    # ---------------------------
    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(x4)

    model = models.Model(inputs=base.input, outputs=outputs, name="UNet_MobileNetV2")

    # ---------------------------
    # 5) COMPILATION (Keras 3)
    # ---------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=dice_loss,
        metrics=[iou_metric, dice_coef, pixel_accuracy]
    )

    return model
