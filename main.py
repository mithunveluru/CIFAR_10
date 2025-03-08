import numpy as np
import tensorflow as tf
import os
from data_prep import load_and_preprocess_data, visualize_examples
from model import build_cnn_model, build_transfer_learning_model
from train import train_model, train_transfer_learning_model
from evaluate import evaluate_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.random.seed(42)
tf.random.set_seed(42)

def main():
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print("Step 1: Loading and preprocessing data")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    print("Step 2: Visualizing example images")
    visualize_examples(x_train, y_train, class_names, output_dir)

    print("Step 3: Building and training CNN model")
    model = build_cnn_model()
    model, history = train_model(model, x_train, y_train, x_val, y_val, output_dir)

    print("Step 4: Evaluating CNN model")
    evaluate_model(model, history, x_test, y_test, class_names, output_dir)

    print("\nWould you like to train a transfer learning model? (y/n)")
    choice = input().strip().lower()

    if choice == 'y':
        print("Step 5: Building and training transfer learning model")
        transfer_model = build_transfer_learning_model()
        transfer_model, transfer_history = train_transfer_learning_model(
            transfer_model, x_train, y_train, x_val, y_val, output_dir
        )

        print("Step 6: Evaluating transfer learning model")
        evaluate_model(
            transfer_model, transfer_history, x_test, y_test, 
            class_names, output_dir + '/transfer_learning'
        )

    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()

