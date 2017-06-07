# Webcam Anomaly Detection

An FTP server that logs anomalous motion from a camera stream.

## Dependencies
* [NumPy](http://www.numpy.org/)
* [Keras](https://keras.io)
* [scikit-image](http://scikit-image.org)
* [pyftpdlib](https://github.com/giampaolo/pyftpdlib)

## What
When working with basic FTP webcam streams, it's easy to end up with disk-fulls of image data.
A basic solution is to delete old footage automatically, but this can end up deleting important events.
Another idea is to analyze the change between every image and only save them if enough pixels changed
indicating motion. While this works, in situations where the camera is positioned in front of trees or towards
clouds, the system will often end up with a plethora of false positives. This project aims to fix this by
using machine learning to determine what is anomalous motion. It does this by creating a delta image which
is pixels of the past image minus the pixels of the current image and classifying it as either noise or an
anomaly. This way only key frames stay while pictures that only show leaves moving are discarded.

## Usage

#### Install
1. ```git clone https://github.com/sshh12/Webcam-Anomaly-Detection.git```
2. ```pip install -r requirements.txt``` (For Keras, [Tensorflow](https://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/) is required)

#### Setup
1. Find a camera that supports FTP logging
2. Create a username and password then add it to the ```config.py```
3. Set ```TRAIN = True``` in ```config.py```
4. Point the camera to the IP (and port 21) of your computer
5. Run ```main.py```, this will populate the training folder with a constant stream of images and delta images (dimages)
6. Once a bunch (>1000) of images have been collected, drag the dimages (\*.d.jpg) into either the anomaly or noise folder
7. Run ```generate_anomaly_model.py``` to generate a model to predict future dimages (this will take a while)
8. Set ```TRAIN = False``` in ```config.py``` and optionally delete the contents of the training folder

#### Running
1. Run ```main.py```
2. Check the images folder to see if images are being saved
3. Close the window or press Ctrl-C to stop the server

### Example Delta Images

#### Anomaly
![Anomaly](https://user-images.githubusercontent.com/6625384/26891644-64de5080-4b7b-11e7-8960-57ce75c99e28.jpg)

#### Noise
![Noise](https://user-images.githubusercontent.com/6625384/26891700-964a0c90-4b7b-11e7-8a69-99525a83bfcb.jpg)
