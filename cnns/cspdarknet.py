import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

# Mish activation function
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

ACTIVATIONS = {
    'mish': Mish(),
    'linear': nn.Identity()
}

# Convolutional block with batch normalization and activation function
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='mish'):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            ACTIVATIONS[activation]
        )

    def forward(self, x):
        return self.conv(x)

# CSP Block used in the network
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, residual_activation='linear'):
        super(CSPBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        self.block = nn.Sequential(
            Conv(in_channels, hidden_channels, 1),
            Conv(hidden_channels, out_channels, 3)
        )

        self.activation = ACTIVATIONS[residual_activation]

    def forward(self, x):
        return self.activation(x + self.block(x))

# First stage of CSPDarknet
class CSPFirstStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirstStage, self).__init__()

        self.downsample_conv = Conv(in_channels, out_channels, 3, stride=2)

        self.split_conv0 = Conv(out_channels, out_channels, 1)
        self.split_conv1 = Conv(out_channels, out_channels, 1)

        self.blocks_conv = nn.Sequential(
            CSPBlock(out_channels, out_channels, in_channels),
            Conv(out_channels, out_channels, 1)
        )

        self.concat_conv = Conv(out_channels*2, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x

# CSP Stage used in the network
class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPStage, self).__init__()

        self.downsample_conv = Conv(in_channels, out_channels, 3, stride=2)

        self.split_conv0 = Conv(out_channels, out_channels//2, 1)
        self.split_conv1 = Conv(out_channels, out_channels//2, 1)

        self.blocks_conv = nn.Sequential(
            *[CSPBlock(out_channels//2, out_channels//2) for _ in range(num_blocks)],
            Conv(out_channels//2, out_channels//2, 1)
        )

        self.concat_conv = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x

# Main CSPDarknet53 model
class CSPDarknet53(nn.Module):
    def __init__(self, stem_channels=32, feature_channels=[64, 128, 256, 512, 1024], num_features=1):
        super(CSPDarknet53, self).__init__()

        self.stem_conv = Conv(3, stem_channels, 3)

        self.stages = nn.ModuleList([
            CSPFirstStage(stem_channels, feature_channels[0]),
            CSPStage(feature_channels[0], feature_channels[1], 2),
            CSPStage(feature_channels[1], feature_channels[2], 8),
            CSPStage(feature_channels[2], feature_channels[3], 8),
            CSPStage(feature_channels[3], feature_channels[4], 4)
        ])

        self.feature_channels = feature_channels
        self.num_features = num_features

    def forward(self, x):
        x = self.stem_conv(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features[-self.num_features:]

# CNN class (base class)
class CNN(nn.Module):
    def __init__(self, number_of_classes):
        super(CNN, self).__init__()
        self.number_of_classes = number_of_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def validate(self, validation_loader, classes, model, model_name):
        model.eval()
        all_preds = []
        all_labels = []
        for images, labels in validation_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = model(images)[-1]  # Get the last output from the feature list
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = sklearn.metrics.accuracy_score(all_labels, all_preds)
        return accuracy, None

# CSPDarknetModel class inheriting from CNN
class CSPDarknetModel(CNN):
    def __init__(self, number_of_classes, model_type="cspdarknet53"):
        super().__init__(number_of_classes)
        self.cnn_name = model_type

        # Load the CSPDarknet model
        if model_type == "cspdarknet53":
            self.cnn_model = CSPDarknet53(num_features=3)
        else:
            raise ValueError("Unsupported model type. Choose 'cspdarknet53'.")

        # Modify the final classifier layer to match the number of classes
        self.fc = nn.Linear(self.cnn_model.feature_channels[-1], self.number_of_classes)
        self.cnn_model = self.cnn_model.to(self.device)

    def fit(self, path_to_dataset, number_of_epochs=10, batch_size=32, learning_rate=0.001, save_the_best_model=False):
        # 1. Setting up transform function: resizing, tensoring, and normalizing.
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Loading datasets (training & validation) and transforming.
        training_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "training"), transform=transform)
        validation_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_to_dataset, "validation"), transform=transform)

        # 3. Creating DataLoaders based on datasets.
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 4. Setting up loss function.
        criterion = torch.nn.CrossEntropyLoss()

        # 5. Setting up optimizer.
        optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=learning_rate)

        # 6. Saving the best validation accuracy.
        best_val_accuracy = 0.0

        # 7. Training loop
        for epoch in range(number_of_epochs):
            self.cnn_model.train()
            running_loss = 0.0

            for images, labels in training_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                # Get the last feature map (outputs[-1]) and apply Global Average Pooling
                outputs = self.cnn_model(images)[-1]
                x = torch.mean(outputs, dim=[2, 3])  # Global average pooling across height and width
                x = self.fc(x)  # Final fully connected layer

                loss = criterion(x, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(training_loader)
            print(f"Epoch {epoch+1}/{number_of_epochs} - Training Loss: {avg_train_loss:.4f}")

            # 8. Validating and evaluating performance after each epoch.
            val_accuracy, val_metrics = self.validate(validation_loader, validation_dataset.classes, self.cnn_model, self.cnn_name)

            # 9. Saving the best model based on validation accuracy.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_the_best_model:
                    torch.save(self.cnn_model.state_dict(), "best_cspdarknet.pth")
                print(f"Best model saved with Accuracy: {best_val_accuracy:.4f}")

        print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")