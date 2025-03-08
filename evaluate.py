import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, history, x_test, y_test, class_names, output_dir='.'):
    print("Evaluating the model...")
    
    # Plot training & validation accuracy and loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png')
    plt.close()
    print(f"Training history saved as '{output_dir}/training_history.png'")
    
    # Evaluate on test data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Get predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Generate classification report
    report = classification_report(y_true_classes, y_pred_classes, 
                                  target_names=class_names, digits=4)
    print("Classification Report:")
    print(report)
    
    # Save classification report to file
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved as '{output_dir}/confusion_matrix.png'")
    
    # Plot sample predictions
    plot_sample_predictions(x_test, y_true_classes, y_pred_classes, class_names, output_dir)

def plot_sample_predictions(x, y_true, y_pred, class_names, output_dir='.', n=25):
    plt.figure(figsize=(10, 10))
    # Choose random samples
    idx = np.random.choice(range(len(x)), n, replace=False)
    
    for i in range(n):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[idx[i]])
        
        true_label = class_names[y_true[idx[i]]]
        pred_label = class_names[y_pred[idx[i]]]
        color = 'green' if y_true[idx[i]] == y_pred[idx[i]] else 'red'
        
        plt.xlabel(f"{pred_label} ({true_label})", color=color)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sample_predictions.png')
    plt.close()
    print(f"Sample predictions saved as '{output_dir}/sample_predictions.png'")
