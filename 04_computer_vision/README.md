# Lesson 04: Computer Vision

In this lesson, you'll learn how to train a Computer Vision model using *pytorch*. Computer vision works on ingesting images or videos to make decisions (segmentations, classifications, etc.). Everything that can be consumed in a "visual sense" is relevant for be used within computer vision.

For computer vision, there are some very important concepts you need to know:

## Tensor Format(RGB)

In the example of image computer vision, tensors are commonly arranged in 3 2D tensors containing the channels for Red, Blue and Green tones.

Take an example of a black & white image that can be shown in a matrix with a number 1:

[0,0,0,1,1,0]
[0,0,1,1,1,0]
[0,0,1,1,1,0]
[0,0,0,1,1,0]
[0,0,0,1,1,0]
[0,0,0,1,1,0]
[0,1,1,1,1,1]
[0,1,1,1,1,1]

This matrix would represent a number 1, where the "number" is related to the intensity of "black" in the image. In computer vision, you can also work with 3 levels of intensity (RGB) where we would have three matrices similar to the one above for each channel.

## Convolutional neural networks

Convolutional Neural Networks (CNNs) are a class of deep neural networks primarily used for analyzing visual imagery. They're designed to automatically and adaptively learn spatial hierarchies of features from the input images through a process called convolution.

**Convolutional Layers:** The core building blocks of CNNs are convolutional layers. Each layer consists of a set of learnable filters (also called kernels) that are convolved with the input image to produce feature maps. These filters detect specific patterns or features, such as edges, textures, or shapes, at different spatial locations in the input image. Convolutional layers apply these filters across the entire input image to extract relevant features.

**Pooling Layers:** Pooling layers are often used in CNNs to reduce the spatial dimensions of the feature maps produced by the convolutional layers, while retaining the most important information. Max pooling, for example, downsamples the feature maps by taking the maximum value within a small region, effectively reducing the size of the feature maps and the computational complexity of the network.

**Activation Functions:** Activation functions, such as ReLU (Rectified Linear Unit), are applied element-wise to the output of convolutional and pooling layers to introduce non-linearity into the network. This enables CNNs to learn complex relationships and representations in the data beyond simple linear transformations.

**Fully Connected Layers:** Towards the end of the CNN architecture, one or more fully connected layers are often added. These layers take the high-level features extracted by the convolutional and pooling layers and use them to classify the input image into different classes or categories. Fully connected layers perform classification based on the learned features, using techniques such as softmax activation to produce probability distributions over the possible classes.

**Transfer Learning:** CNNs trained on large-scale datasets, such as ImageNet, can be adapted to new tasks with limited labeled data using transfer learning. By fine-tuning the pre-trained CNN on a smaller dataset related to the target task, one can leverage the learned feature representations from the original task and achieve good performance with less data and computational resources.

CNNs have demonstrated remarkable success in a wide range of computer vision tasks.

To help us understand the Convolutional Operation, we have our friends at [3Blue1Brown](https://www.youtube.com/watch?v=KuXjwB4LzSA&pp=ygUcY29udm9sdXRpb25hbCBuZXVyYWwgbmV0d29yaw%3D%3D)

And this video from [Stanford University](https://www.youtube.com/watch?v=DAOcjicFr1Y&pp=ygUcY29udm9sdXRpb25hbCBuZXVyYWwgbmV0d29yaw%3D%3D) explains CNN architecture quite well!

## Different CNNs to Rule Them All

Although parameters for CNNs are normally a hyperparameter, there are many typical CNN architectures you can use: 

**VGG19:** [This architecture](https://www.researchgate.net/figure/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means_fig2_325137356) is extremely famous and achieved interesting results. Let's see the architecture: 

![CNNArchitecture](https://www.researchgate.net/publication/325137356/figure/fig2/AS:670371271413777@1536840374533/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means.jpg)

**AlexNet** [This architecture](https://en.wikipedia.org/wiki/AlexNet) competed in ImageNet Large Scale Visual Recognition and achieved impressive performance in 2012. It's one of the largest examples of Convolutional Architects:

![CNNArchitecture](https://pub.mdpi-res.com/remotesensing/remotesensing-09-00848/article_deploy/html/images/remotesensing-09-00848-g001.png?1569499335)

Can you find other examples of commonly used architectures of NNs?

## Extra Resources

If you wish to to explore the part about loading images from a folder, which you didn't to in the article above, check out this [Learn PyTorch Tutorial](https://www.learnpytorch.io/03_pytorch_computer_vision/) on this topic, altough some of the things you may find redundant with the article above. Besides [the following article](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939) also covers the convolutional neural networks quite well.

Also, take note that convolutional neural networks may get stuck in the training process (because of exploding or vanishing gradients) quite easily. Make sure you try different learning rates if you find that your model is suffering from training staleness.

Don't forget to check [01.Computer Vision.ipynb](04_computer_vision\01. Computer Vision.ipynb) notebook, where you'll be able to see a CV algorithm trained on the famous MNIST dataset.