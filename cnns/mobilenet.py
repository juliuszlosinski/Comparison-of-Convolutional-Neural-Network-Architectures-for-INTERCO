from .cnn import CNN
import torch
import torch.nn as nn
import torchvision
import os
import sklearn.metrics

# MobileNet Class inheriting from CNN.
class MobileNet(CNN):
    def __init__(self, number_of_classes, model_type="mobilenet_v2"):
        # 1. Initializing the base CNN class.
        super().__init__(number_of_classes) 
        self.cnn_name = "MobileNet"

        # 2. Loading pre-trained MobileNetV2 model.
        if model_type == "mobilenet_v2":
            self.cnn_model = torchvision.models.mobilenet_v2(pretrained=True)
        else:
            raise ValueError("Only 'mobilenet_v2' is supported currently.")
        
        # 3. Modifing the final fully connected layer to match the number of classes.
        self.cnn_model.classifier[1] = torch.nn.Linear(self.cnn_model.classifier[1].in_features, self.number_of_classes)
        self.cnn_model = self.cnn_model.to(self.device)

    def fit(self, path_to_dataset, number_of_epochs=10, batch_size=32, learning_rate=0.001, save_the_best_model=False):
        # 1. Setting up transform function: resizing, tensoring and normalizing.
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Loading datasets (training & validation).
        training_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "training"), transform=transform)
        validation_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "validation"), transform=transform)

        # 3. Creating DataLoaders based on datasets (batches, shuffled).
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 4. Setting up loss function, which is Cross Entropy Loss/Categorical Cross Entropy Loss.
        criterion = torch.nn.CrossEntropyLoss()
        
        # 5. Setting up Adam optimizer in order to update weights.
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
                outputs = self.cnn_model(images)    # Getting output from vision model.
                loss = criterion(outputs, labels)   # Getting loss result based on output and expected labels.
                loss.backward()                     # Calculating gradients.
                optimizer.step()                    # Using gradients and Adam optimizer in order to update weights.

                running_loss += loss.item()

            avg_train_loss = running_loss / len(training_loader)
            print(f"Epoch {epoch+1}/{number_of_epochs} - Training Loss: {avg_train_loss:.4f}")

            # 8. Validating and drawing results after each epoch.
            val_accuracy, val_metrics = self.validate(validation_loader, validation_dataset.classes, self.cnn_model, self.cnn_name)

            # 9. Saving best model based on validation accuracy.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_the_best_model:
                    torch.save(self.cnn_model.state_dict(), "best_mobilenet.pth")
                print(f"Best model saved with Accuracy: {best_val_accuracy:.4f}")
        print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")