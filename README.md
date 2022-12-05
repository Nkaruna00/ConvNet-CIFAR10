# ConvNet-CIFAR10

Creation of a neural network capable of recognizing objects in an image. Trained with the CIFAR 10 dataset.
The code is written in Python and use the Keras and Tensorflow frameworks.


## Description

The 10 different classes represent : 
* airplanes
* cars
* birds
* cats
* deer
* dogs
* frogs
* horses
* ships
* trucks. 

There are 6,000 images of each class.
The data is augmented with the help of data augmentation tools to get more training data using rotations and flips among others.

The model consists of 5 layers.

* The first layer consists of a convolution layer with Weight Decay, also called L2 Regularization. The activation function used is the rectifier (ReLU).
Batch normalization is applied to improve the stability  

* The second layer consists of a convolution layer with L2 Regularization. The activation function used is the rectifier (ReLU).
Batch normalization is applied to improve the stability.
MaxPooling is applied to downsample the input along its spatial dimensions.
We also apply a Dropout of 0.2 which randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.  

* The third layer consists of a convolution layer with L2 Regularization. The activation function used is (ReLU) with Batch Normalization.  

* The fourth layer consists of a convolution layer with L2 Regularization. The activation function used is (ReLU).
We apply Batch Normalization to improve the stability.
MaxPooling is applied to downsample the input along its spatial dimensions.
We also apply a Dropout of 0.3 which randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.  

* The last layer is composed of 10 neurons (one for each object class.
The activation function used is softmax.  

* The optimizer used is Adam.
* The model reaches an accuracy of 81%.

* The cifar10.h5 contains the trained model with all the weights.

## Getting Started

### Dependencies

* Python
* Keras with Tensorflow Backend

### Executing program

* Run cifar10.py with Python
```
python cifar10.py
```

## Author

KARUNAKARAN Nithushan


## License

This project is licensed under the MIT License - see the LICENSE.md file for details

