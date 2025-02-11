from .cnn import CNN
import torch
import torch.nn as nn
import torchvision
import os
import sklearn.metrics

# GoogLeNet (Inception) Class inheriting from CNN
class Inception(CNN):
    def __init__(self, number_of_classes):
        # 1. Intializing the base class.
        super().__init__(number_of_classes)  
        self.cnn_name = "Inception_v3"

        # 1. Loading pre-trained GoogLeNet/Inception V3 and modifing the last layer.
        self.cnn_model = torchvision.models.inception_v3(pretrained=True)
        self.cnn_model.AuxLogits.fc = torch.nn.Linear(768, self.number_of_classes)  # For auxiliary classifier
        self.cnn_model.fc = torch.nn.Linear(2048, self.number_of_classes)  # Modify final classifier
        self.cnn_model = self.cnn_model.to(self.device)

    def fit(self, path_to_dataset, number_of_epochs=10, batch_size=32, learning_rate=0.001, save_the_best_model=False):
        # 1. Creating transform function: resizing, to tensoring, normalizing.
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((299, 299)),  # GoogLeNet/Inception V3 needs 299x299 input image.
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Loading datasets (training & validation).
        training_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "training"), transform=transform)
        validation_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "validation"), transform=transform)

        # 3. Creating DataLoaders (batches, shuffled).
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 4. Setting up loss function for model, which is Cross Entropy Loss/Categorical Cross Entropy Loss.
        criterion = torch.nn.CrossEntropyLoss()
        
        # 5. Setting up optimizer Adam in order to update weights in specific fashion (learning rate).
        optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=learning_rate)

        # 6. Saving the best validation accuracy.
        best_val_accuracy = 0.0

        # 7. Training loop
        for epoch in range(number_of_epochs):
            self.cnn_model.train()
            running_loss = 0.0

            for images, labels in training_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()                          # Zeroing previous gradients.
                outputs = self.cnn_model(images)               # Forward pass.
                if isinstance(outputs, tuple):                 # Checking if auxiliary outputs exist.
                    outputs, aux_outputs = outputs
                    loss1 = criterion(outputs, labels)         # Loss for main classifier.
                    loss2 = criterion(aux_outputs, labels)     # Loss for auxiliary classifier.
                    loss = loss1 + 0.4 * loss2                 # Combined loss.
                else:
                    loss = criterion(outputs, labels)          # Using normal loss if there is not auxiliary surface.
                loss.backward()                                # Computing gradients.
                optimizer.step()                               # Updating weights.

            avg_train_loss = running_loss / len(training_loader)
            print(f"Epoch {epoch+1}/{number_of_epochs} - Training Loss: {avg_train_loss:.4f}")

            # 8. Validating and drawing results after each epoch.
            val_accuracy, val_metrics = self.validate(validation_loader, validation_dataset.classes, self.cnn_model, self.cnn_name)

            # 9. Saving the best model based on validation accuracy.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_the_best_model:
                    torch.save(self.cnn_model.state_dict(), "best_inception.pth")
                print(f"Best model saved with Accuracy: {best_val_accuracy:.4f}")
        print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")