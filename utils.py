# ==============================================
# utils.py  |  Helper functions for Super-Resolution
# ==============================================

%%writefile utils.py
import tensorflow as tf
import tensorflow_datasets as tfds

# --- Define Image Dimensions ---
HR_SIZE = (128, 128)
LR_SIZE = (HR_SIZE[0] // 2, HR_SIZE[1] // 2)  # 64x64


def preprocess_image(data):
    """
    Normalizes and resizes images for the dataset.
    The model's input will be a bicubic-upscaled version of the LR image.
    """
    # Normalize pixel values to [0, 1]
    hr_image = tf.cast(data['hr'], tf.float32) / 255.0
    lr_image = tf.cast(data['lr'], tf.float32) / 255.0

    # Resize to target dimensions
    hr_image = tf.image.resize(hr_image, HR_SIZE, method='bicubic')
    lr_image = tf.image.resize(lr_image, LR_SIZE, method='bicubic')

    # Create model input by upscaling the low-res image
    model_input_image = tf.image.resize(lr_image, HR_SIZE, method='bicubic')

    return model_input_image, hr_image


def load_div2k_data(batch_size=16):
    """
    Loads the DIV2K dataset and creates an efficient tf.data pipeline.
    Uses the 'bicubic_x2' version for 2x super-resolution.
    """
    print("Loading and preparing DIV2K dataset...")

    # Load the dataset using TensorFlow Datasets
    (train_ds, valid_ds), ds_info = tfds.load(
        'div2k/bicubic_x2',
        split=['train', 'validation'],
        as_supervised=False,  # We provide our own preprocessing
        with_info=True
    )

    # --- Create the Training Pipeline ---
    train_dataset = (
        train_ds
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()                     # Cache for performance
        .shuffle(buffer_size=100)
        .batch(batch_size)
        .repeat()                    # Repeat for multiple epochs
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # --- Create the Validation Pipeline ---
    validation_dataset = (
        valid_ds
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    print("✅ Dataset pipelines created successfully.")
    return train_dataset, validation_dataset, ds_info
