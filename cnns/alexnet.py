from .cnn import CNN
import torch
import torch.nn as nn
import torchvision
import os
import sklearn.metrics

# Inheriting from CNN, AlexNet model
class AlexNet(CNN):
    def __init__(self, number_of_classes):
         # 1. Initialize base class with the number of classes
        super().__init__(number_of_classes) 
        self.cnn_name = "AlexNet"

        # 2. Loading pre-trained AlexNet and modify the final layer
        self.cnn_model = torchvision.models.alexnet(pretrained=True)
        self.cnn_model.classifier[6] = torch.nn.Linear(4096, self.number_of_classes)
        self.cnn_model = self.cnn_model.to(self.device)

    def fit(self, path_to_dataset, number_of_epochs=10, batch_size=32, learning_rate=0.001, save_the_best_model=False):
        # 1. Setting up transform function.
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Loading and transforming training and validation images.
        training_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "training"), transform=transform)
        validation_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "validation"), transform=transform)
        
        # 3. Converting images to loaders with batches and shuffles.
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 4. Loss function for classification.
        criterion = torch.nn.CrossEntropyLoss()
        
        # 5. Optimizer Adam for updating weights.
        optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=learning_rate)

        # 6. Saving the best validation accuracy.
        best_val_accuracy = 0.0
        
        # 7. Training model
        for epoch in range(number_of_epochs):
            self.cnn_model.train()
            running_loss = 0.0

            for images, labels in training_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad() # Zeroing gradients.
                outputs = self.cnn_model(images) # Getting outputs.
                loss = criterion(outputs, labels) # Getting loss metric.
                loss.backward()  # Calculating gradients.
                optimizer.step() # Updating weights by using Adam and calucalted gradients.

                running_loss += loss.item()

            avg_train_loss = running_loss / len(training_loader)
            print(f"Epoch {epoch+1}/{number_of_epochs} - Training Loss: {avg_train_loss:.4f}")

            # Validating, printing and drawing results curves.
            val_accuracy, val_metrics = self.validate(validation_loader, validation_dataset.classes, self.cnn_model, self.cnn_name)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_the_best_model:
                    torch.save(self.cnn_model.state_dict(), "best_alexnet.pth")
                print(f"Best model saved with Accuracy: {best_val_accuracy:.4f}")
        print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")