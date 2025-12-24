# ==============================================
# model.py  |  Residual Super-Resolution Model
# ==============================================


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, Activation


def psnr(y_true, y_pred):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) metric.
    """
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def build_enhanced_model(input_shape=(32, 32, 3)):
    """
    Builds an enhanced residual model for image super-resolution.
    """
    # --- Input Layer ---
    inputs = Input(shape=input_shape)

    # --- Feature Extraction Layers ---
    # Using smaller 3x3 kernels improves efficiency and generalization
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)  # Extra depth for richer features

    # --- Reconstruction Layer ---
    x = Conv2D(3, (3, 3), padding='same')(x)  # No activation here (linear output)

    # --- Residual Connection ---
    # The model learns to predict the missing details (residuals)
    outputs = Add()([inputs, x])
    outputs = Activation('sigmoid')(outputs)  # Keeps pixel values in [0, 1]

    # --- Build and Compile the Model ---
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=[psnr]
    )

    return model

