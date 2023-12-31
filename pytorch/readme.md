# Computer Vision Fellowship - PyTorch Project

Welcome to the second project for the Computer Vision Fellowship! In this project,
we will explore using PyTorch, an industry and research standard framework for
machine learning tasks. You will see how to implement and train the models you
wrote in the first project using PyTorch, and also (optionally) implement a more
advanced Convolutional Neural Network for the ImageNET dataset!

## Project Tasks

In this project, we will implement various neural networks, and by the end even
implement a state-of-the-art Convolutional Neural Network!

### Step Zero

Make sure you have completed all the setup steps listed in the main readme. Then
run `python3 backend.py` to test which backend to use, which will tell you what
device string to use. Make a note of this device string for later.

### Step One

We will start out by re-implementing our iris network, to get you familiar with
PyTorch syntax. Follow the steps in `iris.ipynb`.

### Step Two

Now that you have gotten acquainted with how PyTorch works, we will now construct
a network that previously pushed the limits of our micrograd system: MNIST digit
recognition. Complete the `mnist.ipynb` notebook.

### Step Three

Finally, we can touch on some computer vision techniques that are truly state
of the art. We will implement an EfficientNet, a modern CNN that is in use in
many real-world deployments, from phones to self-driving cars. Follow the steps
listed in `cnn.ipynb`.
