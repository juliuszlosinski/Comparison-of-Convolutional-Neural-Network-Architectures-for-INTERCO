from cnns.alexnet import AlexNet
from cnns.vgg import VGG
from cnns.inception import Inception
from cnns.resnet import ResNet
from cnns.resnext import ResNeXt
from cnns.mobilenet import MobileNet
from cnns.efficientnet import EfficientNet
from cnns.cspnet import CSPNet
from cnns.convnext import ConvNeXt
from cnns.cspdarknet import CSPDarknetModel
import argparse
import warnings

# Disabling all warnings
warnings.filterwarnings("ignore")

# To test:
# - AlexNet (2012) (5 conv + 3 linear), (1)
# - VGG (2014) (Deeper architecture):
#   - VGG 16 (13 Conv + 3 Linear), (2)
#   - VGG 19 (16 Conv + 3 Linear). (3)
# - GoogLeNet/Inception (Inception modules) (2014):
#   - Inception v3 (2015) (smaller Conv). (4)
# - ResNet (2015) (Skip connections):
#   - ResNet-18 (18 layers deep), (5)
#   - ResNet-34 (34 layers deep), (6)
#   - ResNet-50 (50 layers deep, most commonly used) (7).
# - ResNeXt (2017) (Grouped convolutions):
#   - resnext50_32x4d (32 groups per 4 conv) (8).
# - MobileNet (2017) (depthwise seperable conv, pointwise operations):
#   - MobileNetV2 (2018) (invered residual blocks, linear bottlenecks) (9),
# - EfficientNet (2019) (compound scaling method, mobile friendly):
#   - EfficientNet-B0 (10),
#   - EfficientNet-B1 (11).
# - CSPNet (2020) (cross-stage partial connections) (12),
# - ConvNeXt (2020) (transformer inspired architecture) (13).

CNN_TYPES = [
    "alexnet",
    "vgg16",
    "vgg19",
    "inceptionv3",
    "resnet-18",
    "resnet-34",
    "resnet-50",
    "resnext",
    "mobilenetv2",
    "efficientnet-b0",
    "efficientnet-b1",
    "cspnet",
    "cspdarknet",
    "convnext"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example Argument Parser for a Machine Learning Model")
    parser.add_argument('--cnn_type', type=str, required = True,
                        choices=CNN_TYPES, default='alexnet',
                        help='Specify the model type to use (default: alexnet)'
    )
    parser.add_argument('--path_to_dataset', type=str,
                        default="./maritime-flags-dataset/balanced_two_flags", required=True,
                        help='Specify the path to dataset (default: ./maritime-flags-dataset/balanced_two_flags)')
    parser.add_argument('--n_classes', type=int, 
                        default=2, required=True,
                        help='Specify the number of classes (default: 2)'
    )
    parser.add_argument('--batch_size', type=int, 
                        default=32, required=True,
                        help='Specify the batch size (default: 32)'
    )
    parser.add_argument('--n_epochs', type=int, 
                        default=10, required=True,
                        help='Specify the number of epochs (default: 10)'
    )
    parser.add_argument('--learning_rate', type=float, 
                        default=2, required=True,
                        help='Specify the value of learning rate (default: 0.001)'
    )
    args = parser.parse_args()
    
    cnn_type = args.cnn_type
    number_of_classes = args.n_classes
    path_to_dataset = args.path_to_dataset
    number_of_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    print(f"\nConfiguration:")
    print(f"Selected cnn type: {cnn_type}")
    print(f"Number of classes: {number_of_classes}")
    print(f"Path to dataset: {path_to_dataset}")
    print(f"Number of epochs: {number_of_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}\n")
    
    if cnn_type == "alexnet":
        cnn_model = AlexNet(number_of_classes)
    elif cnn_type == "vgg16":
        cnn_model = VGG(number_of_classes, "vgg16")
    elif cnn_type == "vgg19":
        cnn_model = VGG(number_of_classes, "vgg19")
    elif cnn_type == "inceptionv3":
        cnn_model = Inception(number_of_classes)
    elif cnn_type == "resnet-18":
        cnn_model = ResNet(number_of_classes, "resnet18")
    elif cnn_type == "resnet-34":
        cnn_model = ResNet(number_of_classes, "resnet34")
    elif cnn_type == "resnet-50":
        cnn_model = ResNet(number_of_classes, "resnet50")
    elif cnn_type == "resnext":
        cnn_model = ResNeXt(number_of_classes)
    elif cnn_type == "mobilenetv2":
        cnn_model = MobileNet(number_of_classes)
    elif cnn_type == "efficientnet-b0":
        cnn_model = EfficientNet(number_of_classes, "efficientnet_b0")
    elif cnn_type == "efficientnet-b1":
        cnn_model = EfficientNet(number_of_classes, "efficientnet_b1")
    elif cnn_type == "cspnet":
        cnn_model = CSPNet(number_of_classes)
    elif cnn_type == "cspdarknet":
        cnn_model = CSPDarknetModel(number_of_classes)
    elif cnn_type == "convnext":
        cnn_model = ConvNeXt(number_of_classes)
        
    cnn_model.fit(
        path_to_dataset = path_to_dataset,
        number_of_epochs = number_of_epochs, # 100
        batch_size = batch_size,       # 32
        learning_rate = learning_rate, # 0.001
        save_the_best_model = False
    )