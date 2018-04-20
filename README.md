# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


---
### Overview

<!-- #### 1. Submission includes all required files and can be used to run the simulator in autonomous mode -->
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results

<!-- #### 2. Submission includes functional code -->
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

<!-- #### 3. Submission code is usable and readable -->
The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The function `pipeline()` shows the main steps of the entire flow.


### Model architecture

<!-- #### 1. An appropriate model architecture has been employed -->
The function `build_model()` describes my convolution model. The model is basically the famous Nvidia model for an end-to-end autonomous driving and can be described by kera's easy to understand and self-explanatory APIs.
```python
Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='elu')
Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='elu')
Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='elu')
Conv2D(filters=64, kernel_size=(3, 3), activation='elu')
Conv2D(filters=64, kernel_size=(3, 3), activation='elu')
Flatten()
Dense(units=100, activation='elu', kernel_regularizer=l2(0.001))
Dense(units=50, activation='elu', kernel_regularizer=l2(0.001))
Dense(units=10)
Dense(units=1)
```

The model includes `ELU` activation function to introduce nonlinearity, and the data is normalized in the model using a Keras `lambda` layer before the model pipeline starts. I also used `L2 regularization` for the first two flat layers to overcome overfitting.

Below is the summary of the number of parameters and output shape of each layer using kera's `model.summary()`. No wonder the first flat layer has the maximum number of parameters to tune.
```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0
_________________________________________________________________
lambda_1 (Lambda)            (None, 66, 200, 3)        0
_________________________________________________________________
lambda_2 (Lambda)            (None, 66, 200, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________
```

<!-- #### 2. Attempts to reduce overfitting in the model -->
Even though I experimented with the `dropout` and `maxpooling` layers in the flat layers in order to reduce overfitting, eventually, I did nto include them in the final model, as for the car to drive on the easier lane, it was not essential. However, for more challenging lanes, this may be useful.

<!-- #### 3. Model parameter tuning -->
The model used an `Adam optimizer` with learning rate `1e-4`, which was selected after some experimentation.


### Data processing

Before I fed the training data to my model, I performed three pre-processing steps.
* used kera's `Cropping2D()` function to crop the original image to `(70, 20)`.
* resized the images of original size `(160, 320, 3)` to `(66, 200)` using kera's `Lambda()` function.
* normalized the images again using kera's `Lambda()` function to have mean = 0 and std. deviation of 1.


### Data augmentation

I applied the following three common data augmentation techniques on the training images:
* flip horizontally: primarily used opencv's `cv2.flip()` api
* change brightness: the function `change_brighteness_of_image()` implements this functionality. First, I changes the image to HSV color space, then added some random brightness using `np.random.standard_normal()` function and added `0.25` bias to it to avoid having a complete dark image (even though the probability of having it is very low).

For the above two types, the steering angle remains the same as for the original images.

* rotate image: I rotate only the center images. First I get a random angle by `np.random.standard_normal()` and use it in the rotation matrix `cv2.getRotationMatrix2D()`. Then append the image to the training data with a steering angle calculated by the formula `steering_angle - float(random_angle) / 100`.


### Training and validation

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I used the python generators for generating training and validation data for my model, using a batch size of 16, which was chosen after testing on various values (multiples of 16) and primarily comparing the time trade-offs. Generators take more time to train compared to training in a single batch with all of data, however, they are heavy on memory. The function `img_generator()` does this job.

The data provided by Udacity was divided into training and validation sets using sklearn's `train_test_split()` api with a random state value of 42 and validation set consisting 20% of all data. Note that only the training images were augmented thereafter. To avoid overfitting, I further randomly shuffled before selecting each batch and then again within a batch, keeping corresponding x and y values identical. I used a combination of center lane driving, recovering from the left and right sides of the road, both in training and validation data.

Finally, I used kera's `fit_generator()` api to fit the batch data to my model, while training on training data using training_generator and validating on validation data with respect to `validation loss`. The validation set helped determine if the model was over or under fitting. I experimented with 1 to 5 epochs and 1 seems to work fine for the simple lane.


### Discussion

The final step was to run the simulator to see how well the car was driving around track one. Initially, there were a few spots where the vehicle fell off the track, specially around sharp turns, however, after tuning some aforementioned paramters, I was able to improve the driving behavior in those cases.

At the end of the process, the vehicle is able to drive autonomously at the center of the lane around the track and without leaving the road.

Few things still to improve:
* I did not test whether proportion of images with center lane driving, or recovering from either side to the center was represnted in training and validation data with equal proportion.
* With more complex model, along with dropout, regularization, the model can work better with more challenging videos.
