"""
2.8 Error Analysis
Confusion Matrix and visualization of misclassified samples
"""
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


wandb.init(project="da6401-assignment1", name="2.8_error_analysis")

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("mnist")

# Train best model (same as best from sweep)
class Args:
    pass

args = Args()
args.input_size = 784
args.output_size = 10
args.hidden_layers = 2
args.num_neurons = [128, 128]
args.activation = 'relu'
args.weight_init = 'xavier'

model = NeuralNetwork(args)

# Quick training
n_train = X_train.shape[0]
for epoch in range(10):
    indices = np.random.permutation(n_train)
    for i in range(0, n_train, 32):
        batch_indices = indices[i:i+32]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        
        y_pred = model.forward(X_batch)
        model.backward(y_batch, y_pred)
        model.update_weights(learning_rate=0.001)

# Make predictions
y_pred_probs = model.forward(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test_labels, y_pred)

# Plot CM
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on Test Set')
for i in range(10):
    for j in range(10):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.xticks(range(10))
plt.yticks(range(10))
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.close()

# Log to W&B
wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})

# Show misclassified examples
misclassified_idx = np.where(y_test_labels != y_pred)[0]

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    if i < len(misclassified_idx):
        idx = misclassified_idx[i]
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True:{y_test_labels[idx]} Pred:{y_pred[idx]}', fontsize=8)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.savefig('misclassified_examples.png', dpi=100)
plt.close()

wandb.log({"misclassified_examples": wandb.Image('misclassified_examples.png')})

# Accuracy per class
for c in range(10):
    class_mask = y_test_labels == c
    class_acc = np.mean(y_test_labels[class_mask] == y_pred[class_mask])
    wandb.log({f"class_{c}_accuracy": class_acc})

wandb.finish()