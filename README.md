This paper invistages the problem of choosing the best CNN architecture for INTERCO problem.
=======
This study search for the best classification CNN based network for INTERCO flags problem. The best CNN architecture will replace the backbone in YOLOv8 medium model in next paper.

**Hypothesis:** Choosing the best CNN architecture improves classification accuracy. 

## 1. UML
TODO

## 2. Project organization
TODO

## 3. Typical CNN architecture
TODO

## 4. CNN architectures 
![image](https://github.com/user-attachments/assets/7db35f08-66f5-4901-a646-547af2b06dc6)
![image](https://github.com/user-attachments/assets/dee1805c-fbd9-4b4d-bad4-6d117756c871)
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

### 4.3 Inception
![image](https://github.com/user-attachments/assets/d2aca0e5-a616-4d92-a464-4191366d7cac)

### 4.4 ResNet
![image](https://github.com/user-attachments/assets/846e1fbd-8245-471e-8ffe-79ffd1d2eedf)

### 4.5 ResNeXt
![image](https://github.com/user-attachments/assets/0cd2a942-5e77-43fd-a50f-48bf29208406)

### 4.6 MobileNet
![image](https://github.com/user-attachments/assets/e950d7d7-4768-4b28-b747-0ba6dcaa9c49)

### 4.7 EfficientNet

### 4.8 CSPNet

### 4.9 ConvNeXt
