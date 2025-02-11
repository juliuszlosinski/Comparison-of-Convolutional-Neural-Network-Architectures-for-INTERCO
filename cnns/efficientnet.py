from .cnn import CNN
import torch
import torch.nn as nn
import torchvision
import os
import sklearn.metrics

# EfficientNet Class inheriting from CNN
class EfficientNet(CNN):
    def __init__(self, number_of_classes, model_type="efficientnet_b0"):
        super().__init__(number_of_classes)  # Initialize the base CNN class
        self.cnn_name = model_type

        # 1. Loading pre-trained EfficientNet model from torchvision
        if model_type == "efficientnet_b0":
            self.cnn_model = torchvision.models.efficientnet_b0(pretrained=True)
        elif model_type == "efficientnet_b1":
            self.cnn_model = torchvision.models.efficientnet_b1(pretrained=True)
        elif model_type == "efficientnet_b2":
            self.cnn_model = torchvision.models.efficientnet_b2(pretrained=True)
        elif model_type == "efficientnet_b3":
            self.cnn_model = torchvision.models.efficientnet_b3(pretrained=True)
        else:
            raise ValueError("Invalid model type. Choose from 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3'.")
        
        # 2. Modify the classifier to match the number of classes.
        in_features = self.cnn_model.classifier[1].in_features
        self.cnn_model.classifier[1] = torch.nn.Linear(in_features, self.number_of_classes)
        
        # 3. Moving the model to the chosen device (CPU or GPU).
        self.cnn_model = self.cnn_model.to(self.device)

    def fit(self, path_to_dataset, number_of_epochs=10, batch_size=32, learning_rate=0.001, save_the_best_model=False):
        # 1. Setting up transform function: resizing, tensoring and normalizing.
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # EfficientNet typically uses 224x224 images.
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Loading datasets (training & validation) and transforming.
        training_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "training"), transform=transform)
        validation_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "validation"), transform=transform)

        # 3. Creating DataLoaders (batches, shuffled).
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 4. Setting loss function, which is Cross Entropy Loss/Categorical Cross Entropy Loss.
        criterion = torch.nn.CrossEntropyLoss()
        
        # 5. Setting up Adam optimizer in order to update weights.
        optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=learning_rate)

        # 6. Saving the best validation accuracy.
        best_val_accuracy = 0.0

        # 7. Training loop.
        for epoch in range(number_of_epochs):
            self.cnn_model.train()
            running_loss = 0.0

            for images, labels in training_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()               # Zeroing previously calculated gradients.
                outputs = self.cnn_model(images)    # Getting output from model.
                loss = criterion(outputs, labels)   # Getting result based on outputs and expected labels.
                loss.backward()                     # Calculating gradients.
                optimizer.step()                    # Updating weights by using Adam optimizer and gradients.

                running_loss += loss.item()

            avg_train_loss = running_loss / len(training_loader)
            print(f"Epoch {epoch+1}/{number_of_epochs} - Training Loss: {avg_train_loss:.4f}")

            # 8. Validating and drawing results after each epoch.
            val_accuracy, val_metrics = self.validate(validation_loader, validation_dataset.classes, self.cnn_model, self.cnn_name)

            # 9. Saving the best model based on validation accuracy.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_the_best_model:
                    torch.save(self.cnn_model.state_dict(), "best_efficientnet.pth")
                print(f"Best model saved with Accuracy: {best_val_accuracy:.4f}")
        print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")