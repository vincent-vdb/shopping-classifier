# README #

This is a FCN-based classifier for MOKA objects.

## Requirements ##

* Keras >= 2.0
* Tested only with Theano >= 0.9 with libgpuarray backend; should work with other backends (including tensorflow)
* Opencv >= 2.4 with Python binding
* PIL (Pillow)

## Usage ##

### Classification ###

In order to classify objects, you will also need trained neural network weights. The default one used is `lastweights-fullnet.h5`; you will also be able to specify one externally with command line args later.

+ In order to classify objets presented to a V4L2-compatible camera, run the `recognize_camera.py` script. (If there are several suitable cameras available you can specify an optional video device number as argument; otherwise device 0 is used.)
+ In order to classify objects in a sequence of images, run the `recognize_images.py` script with a directory as argument.
    - The directory must contain images with names at the format `image-%04d.jpg`, starting at 1 (image-0001.jpg, image-0002.jpg, image-0003.jpg, etc.)


