#train.py

import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Import the new data loader and the existing model builder
from utils import load_div2k_data
from model import build_enhanced_model, psnr

# --- 1. Training Configuration ---
BATCH_SIZE = 16 # Smaller batch size for larger images to fit in GPU memory
EPOCHS = 30     # Fewer epochs, as each one takes longer. Increase for higher quality.

# --- 2. Load the Dataset ---
train_ds, valid_ds, ds_info = load_div2k_data(batch_size=BATCH_SIZE)

# Calculate steps per epoch
steps_per_epoch = ds_info.splits['train'].num_examples // BATCH_SIZE
validation_steps = ds_info.splits['validation'].num_examples // BATCH_SIZE

# --- 3. Build the Model for 128x128 Input ---
INPUT_SHAPE = (128, 128, 3)
model = build_enhanced_model(input_shape=INPUT_SHAPE)
model.summary()

# --- 4. Train the Model ---
print("\nStarting model training on 128x128 images...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_ds,
    validation_steps=validation_steps
)
print("Training finished.")

# --- 5. Save the New Model ---
if not os.path.exists('models'):
    os.makedirs('models')

model_path = 'models/sr_128_model.h5'
model.save(model_path)
print(f"✅ Model for 128x128 saved to {model_path}")

# --- 6. Visualize a Test Result ---
print("\nVisualizing a sample prediction...")
# Get one batch from the validation dataset to visualize
for lr_batch, hr_batch in valid_ds.take(1):
    # Take the first image from the batch
    lr_image = lr_batch[0]
    hr_image = hr_batch[0]

    # Predict
    pred_image = model.predict(tf.expand_dims(lr_image, axis=0))[0]

    # Plot
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(lr_image)
    plt.title('Low-Res Input (128x128 Upscaled)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(tf.clip_by_value(pred_image, 0, 1)) # Clip values to [0,1] for display
    plt.title('AI Super-Resolved Output')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hr_image)
    plt.title('Original High-Resolution')
    plt.axis('off')
    plt.show()