import torch
import torchvision
import os
import sklearn.metrics

class VGG:
    def __init__(self, number_of_classes, model_type='vgg16'):
        self.number_of_classes = number_of_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Choossing the VGG model type (VGG-11, VGG-13, VGG-16, VGG-19, or VGG-16 BN).
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
            raise ValueError(f"Unknown VGG model type: {vgg_type}. Choose from ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_bn']")

        # 2. Modify the last layer (fully connected layer) to match the number of output classes.
        self.cnn_model.classifier[6] = torch.nn.Linear(4096, self.number_of_classes)
        self.cnn_model = self.cnn_model.to(self.device)

    def fit(self, path_to_dataset, 
            number_of_epochs=10, 
            batch_size=32, 
            learning_rate=0.001):
        # 1. Defining data transforms for VGG architecture.
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Loading datasets (training & validation).
        training_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "training"), transform=transform)
        validation_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "validation"), transform=transform)

        # 3. Creating DataLoaders.
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 4. Defining loss function (CES) and optimizer (ADAM).
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=learning_rate)

        # 5. Tracking best validation accuracy.
        best_val_accuracy = 0.0

        # 6. Training loop.
        for epoch in range(number_of_epochs):
            self.cnn_model.train()
            running_loss = 0.0

            for images, labels in training_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.cnn_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 7. Computing average training loss.
            avg_train_loss = running_loss / len(training_loader)
            print("Epoch {}/{} - Training Loss: {:.4f}".format(epoch+1, number_of_epochs, avg_train_loss))

            # 8. Validating after each epoch.
            val_accuracy, val_metrics = self.validate(validation_loader, validation_dataset.classes)

            # 9. Saving best model based on validation accuracy.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.cnn_model.state_dict(), "best_vgg_model.pth")
                print("Best model saved with Accuracy: {:.4f}".format(best_val_accuracy))

        print("Training complete. Best validation accuracy: {:.4f}".format(best_val_accuracy))

    def validate(self, validation_loader, class_names):
        self.cnn_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.cnn_model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 1. Computing metrics.
        accuracy = sklearn.metrics.accuracy_score(all_labels, all_preds)
        precision = sklearn.metrics.precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = sklearn.metrics.recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = sklearn.metrics.f1_score(all_labels, all_preds, average="macro", zero_division=0)

        print("Validation Results:")
        print("  Accuracy: {:.4f}".format(accuracy))
        print("  Precision: {:.4f}".format(precision))
        print("  Recall: {:.4f}".format(recall))
        print("  F1-score: {:.4f}".format(f1))

        return accuracy, (precision, recall, f1)

    def predict(self, image_path):
        self.cnn_model.eval()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = torchvision.io.read_image(image_path).float() / 255.0
        image = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.cnn_model(image)
            prediction = torch.argmax(output, dim=1).item()

        return prediction