import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import prepare_data
from train_alexnet import create_alexnet
from train_vgg16 import create_vgg16
from train_resnet50 import create_resnet50
from train_inceptionv3 import create_inceptionv3

def evaluate_model(model, x_test, y_test, class_names):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    cifar10_dir = '/Users/Yoshua/cnn-research/cifar-10-batches-py'
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(cifar10_dir)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Evaluate AlexNet
    alexnet = create_alexnet()
    alexnet.load_weights('alexnet_weights.keras')
    alexnet_metrics = evaluate_model(alexnet, x_test, y_test, class_names)

    # Evaluate VGG16
    vgg16 = create_vgg16()
    vgg16.load_weights('vgg16_weights.h5')
    vgg16_metrics = evaluate_model(vgg16, x_test, y_test, class_names)

    # Evaluate ResNet50
    resnet50 = create_resnet50()
    resnet50.load_weights('resnet50_weights.h5')
    resnet50_metrics = evaluate_model(resnet50, x_test, y_test, class_names)

    # Evaluate InceptionV3
    inceptionv3 = create_inceptionv3()
    inceptionv3.load_weights('inceptionv3_weights.h5')
    inceptionv3_metrics = evaluate_model(inceptionv3, x_test, y_test, class_names)

    # Store metrics in a dictionary
    metrics = {
        'Model': ['AlexNet', 'VGG16', 'ResNet50', 'InceptionV3'],
        'Accuracy': [alexnet_metrics[0], vgg16_metrics[0], resnet50_metrics[0], inceptionv3_metrics[0]],
        'Precision': [alexnet_metrics[1], vgg16_metrics[1], resnet50_metrics[1], inceptionv3_metrics[1]],
        'Recall': [alexnet_metrics[2], vgg16_metrics[2], resnet50_metrics[2], inceptionv3_metrics[2]],
        'F1-Score': [alexnet_metrics[3], vgg16_metrics[3], resnet50_metrics[3], inceptionv3_metrics[3]]
    }

    # Convert dictionary to a Pandas DataFrame
    df = pd.DataFrame(metrics)

    # Plot performance comparison
    df.plot(x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'], kind='bar')
    plt.title('Performance Comparison of CNN Architectures')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.show()
