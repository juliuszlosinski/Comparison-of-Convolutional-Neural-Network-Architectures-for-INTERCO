**Goal:** Researching the problem of choosing the best CNN architecture for INTERCO classification problem.

This study search for the best classification CNN based network for INTERCO flags problem. The best CNN architecture will replace the backbone in YOLOv8 medium model in next paper.

**Hypothesis:** Choosing the best CNN architecture for INTERCO dataset improves classification accuracy.

**Used CNNs architectures on INTERCO:**
- AlexNet (2012) (5 conv + 3 linear), [1]
- VGG (2014) (Deeper architecture):
  - VGG 16 (13 Conv + 3 Linear), [2]
  - VGG 19 (16 Conv + 3 Linear). [3]
- GoogLeNet/Inception (Inception modules) (2014):
  - Inception v3 (2015) (smaller Conv). [4]
- ResNet (2015) (Skip connections):
  - ResNet-18 (18 layers deep), [5]
  - ResNet-34 (34 layers deep), [6]
  - ResNet-50 (50 layers deep, most commonly used) [7].
- ResNeXt (2017) (Grouped convolutions):
  - resnext50_32x4d (32 groups per 4 conv) [8].
- MobileNet (2017) (depthwise seperable conv, pointwise operations):
  - MobileNetV2 (2018) (invered residual blocks, linear bottlenecks) [9],
- EfficientNet (2019) (compound scaling method, mobile friendly):
  - EfficientNet-B0 [10],
  - EfficientNet-B1 [11].
- CSPNet (2020) (cross-stage partial connections) [12],
- ConvNeXt (2022) (transformer inspired architecture) [13].

## 1. UML
![Comparison-CNN-Arch](https://github.com/user-attachments/assets/c5f42e9d-dbe9-4cc2-8819-27452d1a24bc)

## 2. Project organization
```
├── documentation       <- UML diagrams and configuration
├── cnns                <- Package with CNN architectures
│   ├── __init__.py     <- Package identicator
│   ├── alexnet.py      <- AlexNet (2012) architecture
│   ├── vgg.py          <- VGG (2014) architecture
│   ├── inception.py    <- GoogLeNet/Incpetion (2014) architecture
│   ├── resnet.py       <- ResNet (2015) architecture
│   ├── resnext.py      <- ResNeXt (2017) architecture
│   ├── mobilenet.py    <- MobileNet (2017) architecture
│   ├── efficientnet.py <- EfficientNet (2019) architecture
|   ├── cspnet.py       <- CSPNet (2020) architecture
│   └── convnext.py     <- ConvNeXt (2022) architecture
├── maritime-flags-dataset    <- Source and balanced flags (A-Z)
│   ├── SMOTE_balanced_flags  <- Balanced flags by using SMOTE balancer (total: 26 000)
│   └── two_balanced_flags    <- Balanced two flags (A and B) per 1000 images
├── .gitignore        <- Ignores venv_environment directory to be pushed (VENV)
├── requirements.txt  <- List for venv with all used packages (VENV)
├── classify.py       <- Classification by using CNN architectures (Fitting/Evaluation)
├── test.bat          <- Tesing CNN architectures for Windows platform
├── alexnet_classify_slurm_script     <- AlexNet CNN SLURM (Linux)
├── vgg16_classify_slurm_script       <- VGG16 CNN SLURM (Linux)
├── vgg19_classify_slurm_script       <- VGG19 CNN SLURM (Linux)
├── inceptionv3_classify_slurm_script <- INCEPTION V3 CNN SLURM (Linux)
├── resnet-18_classify_slurm_script   <- ResNet-18 CNN SLURM (Linux)
├── resnet-34_classify_slurm_script   <- ResNet-34 CNN SLURM (Linux)
├── resnet-50_classify_slurm_script   <- ResNet-50 CNN SLURM (Linux)
├── resnext_classify_slurm_script     <- ResNeXt CNN SLURM (Linux)
├── mobilenetv2_classify_slurm_script     <- MobileNetv2 CNN SLURM (Linux)
├── efficientnet-b0_classify_slurm_script <- EfficientNet B0 CNN SLURM (Linux)
├── efficientnet-b1_classify_slurm_script <- EfficientNet B1 CNN SLURM (Linux)
├── cspnet_classify_slurm_script          <- CSPNet CNN SLURM (Linux)
└── convnext_classify_slurm_script        <- ConvNeXt CNN SLURM (Linux)
```

## 3. Typical CNN architecture
![image](https://github.com/user-attachments/assets/7db35f08-66f5-4901-a646-547af2b06dc6)

## 4. CNN architectures 
![image](https://github.com/user-attachments/assets/1cfe345b-592b-4024-92b3-941f5e356ff2)

### 4.1 AlexNet (2012)
![image](https://github.com/user-attachments/assets/cd41ab30-fbe0-4083-bd00-30ff562461cc)

AlexNet is a deep convolutional neural network (CNN) architecture designed by Alex Krizhevsky in 2012. It won the ImageNet. The architecture consits of eight layers: **five convolutional** layers followed by **three fully connected** layers. It uses **ReLU** (_Rectifier Linear Unit_) activation function, regularization **dropout** technique and **max pooling**. It was one the first models that used GPU power/acceleration in order to make deep learning more practical and efficient.

### 4.2 ZFNet (2013)
![image](https://github.com/user-attachments/assets/5fecc46e-47f4-4cc2-bd00-3f1f88ac50e9)

ZFNet (2013) is an improved version of AlexNet, developed by Zeiler and Fergus. It refreshed AlexNet's architecture by adjusting hyperparameters, leading to better performance in image classification. The most important improvements were: **smaller first-layer filters** (from 11x11 to 7x7), **reduced stride size** for linear details, **improved visualization techinques** in orderd to understand how CNNs process images.

### 4.2 VGGNet (2014)
![image](https://github.com/user-attachments/assets/c15fc04f-fc55-4e96-9b41-5133ed8a2b47)

VGGNet (2014) is a CNN architecture developed by the Visual Geometry Group (VGG) at Oxford. It improved upon previous models by using **deeper architecture** with small (3x3) convolutional filters/kernels. The most important features are: **increased depth** (16 or 19 layers) for better feature extraction, **uniform 3x3 filters/kernels** instead of larger ones for improvin learning capacity, **max pooling layers** to reduce spatial dimensions while preserving key features, **fully connected layers** at the end for classification. It has high computational cost. VGGNet comes in two main variants: **VGG-16** contains 16 weight layers (13 convolutional + 3 fully connected layers), **VGG-19** a deeper version with 19 weight layers (16 convolutional + 3 fully connected layers). Both models follow the same desing principle, using only **3x3** convolutional filters/kernels and **max pooling layers**. VGG-19 have more convolutional layers which makes slightly more efficient.

### 4.3 Inception (2014)
![image](https://github.com/user-attachments/assets/d2aca0e5-a616-4d92-a464-4191366d7cac)

GoogLeNet (Inception) (2014) is cnn architecture developed by Google. It introduced the Inception module, which made the network more efficient and reduced computational costs. The most important features are: **inception module** uses multiple filter/kernel sizes (1x1, 3x3, 5x5) in parallel to capture different levels of detail, **1x1 convolutions** reduces dimensionality before expensive operations improving efficiency, **deep network (22 layers)** but computationally optimized, **global average pooling** instead of fully connected layers in order to reduce number of parameters and overfitting. The Inception (GoogLeNet) family has several versions, each improving efficiency and accuracy: Inception v1 (2014) introduced the **inception module** with multiple filter/kernel sizes, Inception v2 (2015) used **batch normalization** and **factorized convolutions** in order to improve training speed, Inception v3 (2015) optimized the network further with **smaller convolutions** (replacing 5x5 filters with two 3x3 filters), Inception v4 (2016) **combined** inception modules with **ResNet** for better performance (called Inception-ResNet).

### 4.4 ResNet (2015)
![image](https://github.com/user-attachments/assets/846e1fbd-8245-471e-8ffe-79ffd1d2eedf)

ResNet (Residual Network) (2015) is a CNN architecture created by Microsoft. It introduced **residual connections (skip connections)** to solve the problem of **vanishing gradients**, allowing much deeper networks to be trained effectively. The most important features are: **residual connections** skip layers in order to allow direct gradient flow preventing degradation in deep networks, **very deep architecture** can have 50/101/152 or even 1000+ layers, **bootleneck** layers (1x1 convolutions) reduce computation while maintaining accuracy, **batch normalization** in order to have stable and faster training. Common ResNet variants: **ResNet-18 & ResNet-34** shallower models for smaller tasks, **ResNet-50** uses bottleneck layers for better efficiency, **ResNet-101 & ResNet-152** for deeper networks for high-accuracy tesks, **ResNeXt** an improved version with grouped convolutions for better efficiency.

### 4.5 ResNeXt (2017)
![image](https://github.com/user-attachments/assets/0cd2a942-5e77-43fd-a50f-48bf29208406)

ResNet**X**t is an advanced version of ResNet, developed by Facebook AI Research. It improves efficiency by introducing **grouped convolutions**, allowing for more flexible and scalable deep networks. The most important features: **cardinality (grouped convolutions)** instead of just stacking layers deeper ResNeXt increases **parallel paths** within layers improving feature extraction, **modular design** uses **split-transform-merge** strategy, similar to Inception (GoogLeNet) but more structured, higher accuracy with fewer parameters compared to ResNet, more efficient than deeper ResNets like ResNet-101 and ResNet-152. ResNeXt produces better performance than ResNet while maintaining a similar number of parameters, making it widel used for large-scale vision tasks.

### 4.6 MobileNet (2017)
![image](https://github.com/user-attachments/assets/e950d7d7-4768-4b28-b747-0ba6dcaa9c49)

MobileNet (2017) is a lightweight CNN architecture/model designed by Google for mobile and **embedded devices**. It optimizes efficiency by using **depthwise separable convolutions**, reducing computational cost while maintaining accuracy. The most important features are: **Depthwise Separable Convolutions** splits standard convolutions into **depthwise (per-channel)** and **pointwise (1x1)** operations reducing computation, **lighweight architecture** optimized for low-power devices like smartphones and internet of things devices, **trade-off between speed and accuracy** using **width multiplier** and **resolution multiplier**. MobileNet variants: **MobileNetV1 (2017)** introduced depthwise separable convolutions for efficiency, **MobileNetV2 (2018)** added **inverted residual blocks** and **linear bottlenecks** for better feature reuse, **MobileNetV3 (2019)** used **NAS (Neural Architecture Search)** for further optimize the model for speed and accuracy.

### 4.7 EfficientNet (2019)
![image](https://github.com/user-attachments/assets/2dc88ade-c9c3-4978-8348-1d4a3e40d1e1)

EfficientNet (2019) is a deep learning model developed by Google AI that focuses on optimizing both accuracy and efficiency. It introduces a **compound scaling method** in order to scale the model in a more balanced way, improving performance while keeping computational costs low. The most important features: **Compound scaling** uniformly scales depth, width and resolution of the network instead of scaling one dimension in ResNet, **mobile-friendly** designed to work well on resource limited devices, **improved architecture** uses **depthwise separable convolutions** and **effiecient use of parameters** enhancing performance per computation. EfficientNet have multiple variants: **EfficientNet-B0 to B7** a family of models, with **B0** being smallest and **B7** the largest. These models allow users to choose between a balance of speed and accuracy based on the task. EfficientNet is known for achieving state-of-the-art (SOTA) performance on various benchmarks while requiring fewer resources than previous models like **ResNet** and **VGG** making it highly popular for both large-scale tasks and mobile applications.

### 4.8 CSPNet (2020)
![image](https://github.com/user-attachments/assets/04035b32-ff52-4acb-8b81-f7aa9de13dcd)

CSPNet (Cross-Stage Partial Network) is a CNN architecture designed to improve the training efficiency and performance. It introduces **cross-stage partial connections**, which enhance **gradient flow** and **feature reuse** across different stages of the network. The primary goal is to enable better performance with fewer computational resources. The most important features are: **cross-stage partial connections** divides the feature maps into two parts, with one part going through the full network and the other partially connected. This improves the **gradient flow** and allows for better reuse, **Improved gradient flow** by splitting and merging feature maps, CSPNet facilitates easier training, helping to prevent issues like vanishing gradients, **efficieny and lightweight** the design allows CSPNet to achieve higher accuracy with fewer parameters, making it computationally efficient. CSPNet have variants: **CSPDarkNet** a variant of DarkNet that incorporates the principles of CSPNet for YOLO-based object detection, **It combines the CSPNet architecture with the YOLO framework** in order to improve detection accuracy and reduce computational load. **CSPResNet** integrates the CSPNet design with **ResNet**, offering the benefits of residual connections with enhancing **feature reuse** and **gradient flow**.

### 4.9 ConvNeXt (2022)
![image](https://github.com/user-attachments/assets/b02e8222-4764-43ea-bdef-9cbf1457a057)

ConvNeXt (2022) is a modern CNN architecture developed by Facebook AI Research. It is designed to bridge the gap between the traditional CNNs and the more advanced **transformer-based architectures**, like **Vision Transformers (ViT)**, by introducing **architectural changes** that enhance the performance of CNNs while retaining theif efficiency. ConvNeXt blends the strenghts of CNNs and transformers ideas. The most important features: **transformers-inspired design** borrows key concepts from Vision Transformers (ViT) but remains a CNN, with architectural modifications such as **LayerNorm** and **larger kernel sizes**, **simplified architecture** improves upon the traditional ResNet-style architectures with simpler desings and fewer hyperparameters, making it easier to train and scale. **High performance** it achieves SOTA results on several benchmarks like **ImageNet** by using modern techniques (such as **larger patch sizes**, **relative positional encoding** and **depthwise convolutions**). **Scalability** is designed to be scalable, and its performance improves across model sizes (small to large). The ConvNeXt types: **ConvNeXt-Tiny** a lightweight version with fewer parameters and computation, designed for real-time application and resource-constrained environments. **ConvNext-Small** a sligthly larger model, balancing accuracy and efficiency, suitable for more powerful but still resource-limited applications. **ConvNeXt-Base** the standard model, which achieves competitive performance and is often used for a variety of general-purpose tasks. **ConvNeXt-Large** a larger version for high-accuracy tasks, designed for powerful GPUs or cloud-based systems where computational resources are not as constrained. **ConvNeXt-XLarge** the largest version, offering top-tier accuracy on image classification and related tasks but requiring significant computational resources.
