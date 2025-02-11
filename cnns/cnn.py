import torch
import torchvision
import os
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

# Base class CNN
class CNN:
    def __init__(self, number_of_classes, device=None):
        self.number_of_classes = number_of_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Validation method common for all models
    def validate(self, validation_loader, class_names, cnn_model, cnn_name):
        cnn_model.eval()
        all_preds = []
        all_labels = []
        all_probs = []  # Store probabilities for PR curve

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = cnn_model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())  # Store probabilities

        accuracy = sklearn.metrics.accuracy_score(all_labels, all_preds)
        precision = sklearn.metrics.precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = sklearn.metrics.recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = sklearn.metrics.f1_score(all_labels, all_preds, average="macro", zero_division=0)

        print("Validation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")

        self.plot_precision_recall_curve(all_labels, np.array(all_probs), class_names, cnn_name)
        self.plot_precision_confidence_chart(all_labels, np.array(all_probs), class_names, cnn_name)

        return accuracy, (precision, recall, f1)
    
    # Plot Precision-Recall Curve
    def plot_precision_recall_curve(self, labels, probs, class_names, cnn_name):
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(class_names):
            precision, recall, _ = sklearn.metrics.precision_recall_curve((np.array(labels) == i).astype(int), probs[:, i])
            plt.plot(recall, precision, label=f"{class_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid()
        plt.savefig(f"{cnn_name}_Precision_recall.png")

    # Plot Precision vs Confidence Threshold
    def plot_precision_confidence_chart(self, labels, probs, class_names, cnn_name):
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(class_names):
            sorted_probs = np.sort(probs[:, i])[::-1]
            precisions = [sklearn.metrics.precision_score((np.array(labels) == i).astype(int), probs[:, i] >= t, zero_division=0) for t in sorted_probs]
            plt.plot(sorted_probs, precisions, label=f"{class_name}")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Precision")
        plt.title("Precision vs Confidence")
        plt.legend()
        plt.grid()
        plt.savefig(f"{cnn_name}_Precision_confidence.png")

    # Prediction function common to all models
    def predict(self, image_path, cnn_model):
        cnn_model.eval()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = torchvision.io.read_image(image_path).float() / 255.0
        image = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = cnn_model(image)
            prediction = torch.argmax(output, dim=1).item()
        return prediction