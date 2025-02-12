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

        # Compute per-class metrics
        class_precisions = sklearn.metrics.precision_score(all_labels, all_preds, average=None, zero_division=0)
        class_recalls = sklearn.metrics.recall_score(all_labels, all_preds, average=None, zero_division=0)
        class_f1s = sklearn.metrics.f1_score(all_labels, all_preds, average=None, zero_division=0)
        class_report = sklearn.metrics.classification_report(all_labels, all_preds, zero_division=0)

        print("Validation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")

        output_metrics = "Per-Class Metrics:\n"
        print("\nPer-Class Metrics:")
        for i, (prec, rec, f1) in enumerate(zip(class_precisions, class_recalls, class_f1s)):
            output_metrics = output_metrics + f"  Class {i}: Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}\n"
            print(f"  Class {i}: Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
        
        file = open(f".\\results\{cnn_name}_metrics.txt", "a")
        file.write(output_metrics)
        file.close()    
        
        self.plot_precision_recall_curve(all_labels, np.array(all_probs), class_names, cnn_name)
        self.plot_precision_confidence_chart(all_labels, np.array(all_probs), class_names, cnn_name)

        return accuracy, (precision, recall, f1)
    
    # Plot Precision-Recall Curve
    def plot_precision_recall_curve(self, labels, probs, class_names, cnn_name):
        plt.rcParams["figure.autolayout"] = True
        plt.figure(figsize=(8, 6))
        all_precisions_recall = []
        all_recalls = []

        for i, class_name in enumerate(class_names):
            precision, recall, _ = sklearn.metrics.precision_recall_curve((np.array(labels) == i).astype(int), probs[:, i])
            all_precisions_recall.append(precision)
            all_recalls.append(recall)
            plt.plot(recall, precision, color='gray')
            
        # Compute average precision-recall line
        common_recalls = np.linspace(0, 1, 100)  # Define common recall values
        interp_precisions_recall = [np.interp(common_recalls, all_recalls[i][::-1], all_precisions_recall[i][::-1]) for i in range(len(class_names))]
        avg_precisions_recall = np.mean(interp_precisions_recall, axis=0)
        plt.plot(common_recalls, avg_precisions_recall, color='blue', linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(False)
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.0])
        plt.savefig(f".\\results\{cnn_name}_Precision_recall.png")

    # Plot Precision vs Confidence Threshold
    def plot_precision_confidence_chart(self, labels, probs, class_names, cnn_name):
        plt.rcParams["figure.autolayout"] = True
        plt.figure(figsize=(8, 6))
        all_precisions = []
        all_thresholds = []

        for i, class_name in enumerate(class_names):
            sorted_probs = np.sort(probs[:, i])[::-1]
            precisions = [sklearn.metrics.precision_score((np.array(labels) == i).astype(int), probs[:, i] >= t, zero_division=0) for t in sorted_probs]
            all_precisions.append(precisions)
            all_thresholds.append(sorted_probs)
            plt.plot(sorted_probs, precisions, color="gray")

        # Compute average precision line
        common_thresholds = np.linspace(0, 1, 100)  # Define common thresholds
        interp_precisions = [np.interp(common_thresholds, all_thresholds[i][::-1], all_precisions[i][::-1]) for i in range(len(class_names))]
        avg_precisions = np.mean(interp_precisions, axis=0)
        plt.plot(common_thresholds, avg_precisions, color='blue', linewidth=2)
        plt.xlabel("Confidence")
        plt.ylabel("Precision")
        plt.title("Precision-Confidence Curve")
        plt.grid(False)
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.0])
        plt.savefig(f".\\results\{cnn_name}_Precision_confidence.png")

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