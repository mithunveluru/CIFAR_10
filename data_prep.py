import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data():
    print("Loading and preprocessing the CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Get a subset of the training data for validation
    val_size = 5000
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def visualize_examples(x_train, y_train, class_names, output_dir='.'):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i])
        plt.xlabel(class_names[np.argmax(y_train[i])])
    plt.savefig(f'{output_dir}/cifar10_examples.png')
    plt.close()
    print(f"Example images saved as '{output_dir}/cifar10_examples.png'")

if __name__ == "__main__":
    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load and preprocess data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # Visualize examples
    visualize_examples(x_train, y_train, class_names)
