import wandb
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import load_dataset  # your existing loader
import pandas as pd

# Initialize W&B run
wandb.init(project="mnist_exploration", name="data_exploration")

# Load dataset
_, _, _, _, X_test, y_test = load_dataset("mnist")  # or "fashion_mnist"

# Reshape if flattened
X_test_images = X_test.reshape(-1, 28, 28)

num_samples_per_class = 5
classes = np.argmax(y_test, axis=1)
unique_classes = np.unique(classes)

# Prepare a W&B Table
table = wandb.Table(columns=["Class", "Image"])

for cls in unique_classes:
    idxs = np.where(classes == cls)[0][:num_samples_per_class]
    for idx in idxs:
        image_array = X_test_images[idx]
        table.add_data(cls, wandb.Image(image_array))

# Log to W&B
wandb.log({"sample_images_table": table})
wandb.finish()

print("Logged sample images to W&B successfully!")