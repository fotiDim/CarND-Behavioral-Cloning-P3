# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./examples/model.png "Model Visualization"
[center]: ./examples/center.jpg "Center Image"
[center_cropped]: ./examples/center_cropped.jpg "Center Image Cropped"
[center_cropped_flipped]: ./examples/center_cropped_flipped.jpg "Center Image Flipped"
[left]: ./examples/left.jpg "Left Image"
[left_cropped]: ./examples/left_cropped.jpg "Left Image Cropped"
[left_cropped_flipped]: ./examples/left_cropped_flipped.jpg "Left Image Flipped"
[right]: ./examples/right.jpg "Right Image"
[right_cropped]: ./examples/right_cropped.jpg "Flipped Image Cropped"
[right_cropped_flipped]: ./examples/right_cropped_flipped.jpg "Flipped Image Flipped"
[steering_wheel]: ./examples/steering_wheel.jpg "Steering wheel"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. I used the exact [Nvidia - End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) architecture.

The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

![alt text][model]

My model uses the original [Nvidia architecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) unaltered. Please consult the linked paper for more details.

The only addition was the preprocessing of the data which was done using Lambda layers. I used 2 layers for that:
- Normalizing the pixel values to the range (-0.5, 0.5)
- Cropping the image 70 pixels from above and 25 from the bottom.

#### 2. Attempts to reduce overfitting in the model

I splitted the dataset into: 
- 80% training data
- 20% test data

I used all 3 cameras from the simulator. I augmented the training data by flipping all 3 images as follows:

|   | LEFT | CENTER | RIGHT|
|---|---|---|---|
|original | ![alt text][left]  | ![alt text][center]  | ![alt text][right]  |
|cropped| ![alt text][left_cropped]   | ![alt text][center_cropped]  | ![alt text][right_cropped]  |
|flipped| ![alt text][left_cropped_flipped]   | ![alt text][center_cropped_flipped]  | ![alt text][right_cropped_flipped]  |

The model was trained and validated on the data set to ensure that the model was not overfitting with the loss being close to 0 for the validation set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

5 epochs was enough as evidenced by the minimal loss on my training set.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

#### 4. Appropriate training data

Training data was recorded using `Version 1, 12/09/16` of the [Udacity simulator](https://github.com/udacity/self-driving-car-sim). The reason for using the older version was that version 2 broke support for external peripherals like gamepads. 

I wanted to have precise control for controlling the car while recording so I opted for using a racing wheel. I used a G25 steering wheel from Logitech. You can see my setup here:
![alt text][steering_wheel].

For the left and right images I set a `0.05` steering angle adjustment when training.

#### 3. Creation of the Training Set & Training Process

My driving strategy was to do very smooth handling staying always at the center of the lane even around sharp corners. I drove around 10 laps using only the first track of the simulator and driving always on the same direction.

I augmeneted the data as described above.

After the collection process, I had a little bit more than 10000 data points. I didn't do any manual preprocessing step apart from the lambda layers in the model.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

Training was quite straight forward and the architecture proved to be effective.

#### 4. Caveats 

One big catch, which was not immediately obvious, was that the images read in `model.py` using cv2 were using the BGR format while the images read in the provided `drive.py` used the RGB format by default. The result of this was that the car would initially drive well but eventually would gradually veer out of the first corner. I solved it by converting the images to RGB in `model.py` so the `drive.py` remained intact.
