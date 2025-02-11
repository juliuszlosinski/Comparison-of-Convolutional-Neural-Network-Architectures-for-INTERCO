from .cnn import CNN
import torch
import torch.nn as nn
import torchvision
import os
import sklearn.metrics

# ResNeXt Class inheriting from CNN
class ResNeXt(CNN):
    def __init__(self, number_of_classes, model_type="resnext50_32x4d"):
        # 1. Initializing the base CNN class.
        super().__init__(number_of_classes)  
        self.cnn_name = model_type

        # 2. Loading the pre-trained ResNeXt model based on the specified model type.
        if model_type == "resnext50_32x4d":
            self.cnn_model = torchvision.models.resnext50_32x4d(pretrained=True)
        elif model_type == "resnext101_32x8d":
            self.cnn_model = torchvision.models.resnext101_32x8d(pretrained=True)
        elif model_type == "resnext101_64x4d":
            self.cnn_model = torchvision.models.resnext101_64x4d(pretrained=True)
        elif model_type == "resnext152_32x4d":
            self.cnn_model = torchvision.models.resnext152_32x4d(pretrained=True)
        else:
            raise ValueError("Invalid model type. Choose from 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext152_32x4d'.")

        # 3. Modifing the last fully connected layer to match the number of classes.
        in_features = self.cnn_model.fc.in_features
        self.cnn_model.fc = torch.nn.Linear(in_features, self.number_of_classes)
        self.cnn_model = self.cnn_model.to(self.device)

    def fit(self, path_to_dataset, number_of_epochs=10, batch_size=32, learning_rate=0.001, save_the_best_model=False):
        # 1. Setting up transform function: resizing, tensoring and normalizing.
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Loading datasets (training & validation) and transforming.
        training_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "training"), transform=transform)
        validation_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "validation"), transform=transform)

        # 3. Creating DataLoaders by using datasets.
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 4. Setting up a loss function, which Cross Entropy Loss/Categorical Cross Entropy Loss.
        criterion = torch.nn.CrossEntropyLoss()
        
        # 5. Setting up optimizer in order to update weights.
        optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=learning_rate)

        # 6. Saving the best validation accuracy.
        best_val_accuracy = 0.0

        # 7. Training loop
        for epoch in range(number_of_epochs):
            self.cnn_model.train()
            running_loss = 0.0

            for images, labels in training_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()               # Zeroing previously calculated gradients.
                outputs = self.cnn_model(images)    # Getting output from vision/classifer model.
                loss = criterion(outputs, labels)   # Getting loss result from model.
                loss.backward()                     # Calculating the gradients.
                optimizer.step()                    # Using gradients and optimizer in order to update weights.

                running_loss += loss.item()

            avg_train_loss = running_loss / len(training_loader)
            print(f"Epoch {epoch+1}/{number_of_epochs} - Training Loss: {avg_train_loss:.4f}")

            # 8. Validating and drawing results after each epoch.
            val_accuracy, val_metrics = self.validate(validation_loader, validation_dataset.classes, self.cnn_model, self.cnn_name)

            # 9. Saving the best model based on validation accuracy.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_the_best_model:
                    torch.save(self.cnn_model.state_dict(), "best_resnext.pth")
                print(f"Best model saved with Accuracy: {best_val_accuracy:.4f}")
        print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")