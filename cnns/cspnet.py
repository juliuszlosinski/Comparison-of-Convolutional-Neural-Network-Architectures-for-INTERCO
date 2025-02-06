import torch
import torch.nn as nn
import torchvision
import os
import sklearn.metrics

# Basic CSPNet block
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(CSPBlock, self).__init__()

        self.split = in_channels // 2
        self.conv1 = nn.Conv2d(self.split, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.split, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. Splitting input into two parts.
        x1 = x[:, :self.split, :, :]
        x2 = x[:, self.split:, :, :]

        # 2. Appling convolutions to the two parts.
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        # 3. Concatenating the results.
        x = torch.cat([x1, x2], dim=1)

        # 4. Appling the shortcut connection.
        x = self.conv3(x)
        x = self.shortcut(x)

        return self.relu(x)

# CSPNet model
class CSPNet(nn.Module):
    def __init__(self, number_of_classes):
        super(CSPNet, self).__init__()

        self.number_of_classes = number_of_classes

        # 1. Starting with a pre-trained ResNet backbone for simplicity.
        self.backbone = torchvision.models.resnet18(pretrained=True)  # ResNet-18 as backbone
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone.maxpool = nn.Identity()  # Simplify the maxpooling
        
        # 2. Customizing CSP blocks in the network.
        self.csp_block1 = CSPBlock(64, 64)
        self.csp_block2 = CSPBlock(128, 128)
        self.csp_block3 = CSPBlock(256, 256)
        self.csp_block4 = CSPBlock(512, 512)

        # 3. Replacing fully connected layer to match the number of classes.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, self.number_of_classes)

    def forward(self, x):
        # 1. Passing through the initial layers of the ResNet backbone.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        # 2. CSPNet blocks (modified ResNet structure).
        x = self.csp_block1(x)
        x = self.backbone.layer1(x)

        x = self.csp_block2(x)
        x = self.backbone.layer2(x)

        x = self.csp_block3(x)
        x = self.backbone.layer3(x)

        x = self.csp_block4(x)
        x = self.backbone.layer4(x)

        # 3. Global Average Pooling + Final FC layer.
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x

class CSPNetWrapper:
    def __init__(self, number_of_classes, model_type="cspnet_tiny"):
        self.number_of_classes = number_of_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Loading the simplified CSPNet model.
        self.model = CSPNet(number_of_classes, model_type)
        self.model = self.model.to(self.device)

    def fit(self, path_to_dataset, 
            number_of_epochs=10, 
            batch_size=32, 
            learning_rate=0.001,
            save_the_best_model=False):
        # 1. Defining data transforms
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # CSPNet typically uses 224x224 images
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Loading datasets (training & validation)
        training_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "training"), transform=transform)
        validation_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "validation"), transform=transform)

        # 3. Creating DataLoaders
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 4. Defining loss function (CrossEntropyLoss) and optimizer (Adam)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # 5. Tracking best validation accuracy
        best_val_accuracy = 0.0

        # 6. Training loop
        for epoch in range(number_of_epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in training_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 6.1 Computing average training loss
            avg_train_loss = running_loss / len(training_loader)
            print("Epoch {}/{} - Training Loss: {:.4f}".format(epoch+1, number_of_epochs, avg_train_loss))

            # 6.2 Validating after each epoch
            val_accuracy, val_metrics = self.validate(validation_loader, validation_dataset.classes)

            # 6.3 Saving best model based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_the_best_model:
                    torch.save(self.model.state_dict(), "best_cspnet.pth")
                print("Best model saved with Accuracy: {:.4f}".format(best_val_accuracy))

        print("Training complete. Best validation accuracy: {:.4f}".format(best_val_accuracy))

    def validate(self, validation_loader, class_names):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 1. Computing metrics
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
        self.model.eval()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = torchvision.io.read_image(image_path).float() / 255.0
        image = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            prediction = torch.argmax(output, dim=1).item()

        return prediction