# Computer Vision Fellowship - Micrograd Project

Welcome to the first project for the Computer Vision Fellowship! We will be
writing an autograd engine from scratch, based heavily off of Andrej Karpathy's
micrograd engine. While this sounds complicated, the total engine will only be
roughly 150 lines of code! Then, using this engine, we will create a neural
network that can classify handwritten digits from the MNIST dataset.

## Knowledge Required

While this project is relatively short and simple, it requires an understanding
of a few ideas from math and computer science. Here's a list of things you may
need to brush up on:

* (Multivariable) Calculus, specifically derivatives, gradients, and the chain rule.
* Linear Algebra, namely matrix multiplication
* Object Oriented Programming (in Python), specifically operating overloading/dunder methods.

Also, you should know what a neural network is and understand gradient descent.

## Your Task

The goal of this project is to train a fully-connected, feed-forward neural
network, completely from scratch, to recognize handwritten digits from the MNIST
dataset.

### Step Zero

Make sure you have completed all the setup steps listed in the main readme.

### Step One

First, we need to get `micrograd` up and running, so that you can use it
for your MNIST classifier. Thus, you must complete the code inside the
`micrograd` directory:

Inside the `micrograd` subfolder, you can find the python module we will be
writing. This folder contains three files:
* `__init__.py` is empty, and need not be changed. This is so python will recognize the other files as a module.
* `engine.py` is where our base autograd engine code will go. 
* `nn.py` is where we will put some helper functions that will make it easier for us to create neural networks.

To complete this project, you will need to edit `micrograd/engine.py` and
`micrograd/nn.py`, completing the skeleton that has been given to you in these
files. You can test your work by running `python -m pytest`.

### Step Two

Now that we have a working framework, we can create our neural network. Open
`mnist.ipynb` by running `jupyter notebook` (or open it in VSCode).
