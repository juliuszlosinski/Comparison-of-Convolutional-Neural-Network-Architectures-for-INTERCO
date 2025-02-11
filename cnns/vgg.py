from .cnn import CNN
import torch
import torch.nn as nn
import torchvision
import os
import sklearn.metrics

# VGG Class inheriting from CNN
class VGG(CNN):
    def __init__(self, number_of_classes, model_type='vgg16'):
        # 1. Initializing the base class.
        super().__init__(number_of_classes) 
        self.cnn_name = model_type

        # 2. Choosing the VGG model type (VGG11 - VGG19).
        if model_type == 'vgg11':
            self.cnn_model = torchvision.models.vgg11(pretrained=True)
        elif model_type == 'vgg13':
            self.cnn_model = torchvision.models.vgg13(pretrained=True)
        elif model_type == 'vgg16':
            self.cnn_model = torchvision.models.vgg16(pretrained=True)
        elif model_type == 'vgg19':
            self.cnn_model = torchvision.models.vgg19(pretrained=True)
        elif model_type == 'vgg16_bn':
            self.cnn_model = torchvision.models.vgg16_bn(pretrained=True)
        else:
            raise ValueError(f"Unknown VGG model type: {model_type}. Choose from ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_bn']")

        # 3. Modifing the last fully connected layer to match the number of output classes.
        self.cnn_model.classifier[6] = torch.nn.Linear(4096, self.number_of_classes)
        self.cnn_model = self.cnn_model.to(self.device)

    def fit(self, path_to_dataset, number_of_epochs=10, batch_size=32, learning_rate=0.001, save_the_best_model=False):
        # 1. Creating transform function: resizing, tensoring and normalizing.
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Loading datasets (training & validation) and transforming.
        training_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "training"), transform=transform)
        validation_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "validation"), transform=transform)

        # 3. Creating DataLoaders (batches) from datasets.
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 4. Setting up loss function, which is Cross Entropy Loss/ Categorical Cross Entropy Loss.
        criterion = torch.nn.CrossEntropyLoss()
        
        # 5. Setting up optimizer in order updates weights.
        optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=learning_rate)

        # 6. Saving the best validation accuracy.
        best_val_accuracy = 0.0

        # 7. Training loop
        for epoch in range(number_of_epochs):
            self.cnn_model.train()
            running_loss = 0.0

            for images, labels in training_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()               # Zeroing previously cacluated gradients.
                outputs = self.cnn_model(images)    # Getting outputs from model.
                loss = criterion(outputs, labels)   # Getting loss result from Cross Entropy Loss function.
                loss.backward()                     # Backwarding in order to calculate gradients.
                optimizer.step()                    # Using gradients to update weights.

                running_loss += loss.item()

            avg_train_loss = running_loss / len(training_loader)
            print(f"Epoch {epoch+1}/{number_of_epochs} - Training Loss: {avg_train_loss:.4f}")

            # 8. Validating and drawing after each epoch.
            val_accuracy, val_metrics = self.validate(validation_loader, validation_dataset.classes, self.cnn_model, self.cnn_name)

            # 9. Save best model based on validation accuracy.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_the_best_model:
                    torch.save(self.cnn_model.state_dict(), "best_vgg_model.pth")
                print(f"Best model saved with Accuracy: {best_val_accuracy:.4f}")
        print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")