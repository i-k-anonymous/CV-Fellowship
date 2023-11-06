# AI Safety at UCLA Intro Fellowship: Computer Vision

Last updated: Fall 2023 by Christopher Milan

This is the monorepo for AI Safety at UCLA's Computer Vision Intro Fellowship!

## Introduction

In this fellowship, we will both explore concepts intrinsic to the AI Safety
discussion, from the alignment problem to X-Risk analysis, while also providing
technical background in the field of machine learning by exploring the subfield
of computer vision. 

This repository serves as the center hub of materials for the latter. Your
facilitator should have sent you the materials for the former separately.

## Structure

The goal on the technical side of this fellowship is for all fellows to have
implemented a model capable of classifying the MNIST dataset, both from scratch
(based on Karpathy's [micrograd](https://github.com/karpathy/micrograd)) and
also to introduce the PyTorch framework by porting this model. Thus, the repo is
split into two separate folders, first the `micrograd` project, and secondly the
`pytorch` project. 

On the `master` branch of this repository, empty versions of the homeworks are
provided, while on the `solutions` branch, there are completed versions. Fellows
are encouraged to do their best to solve the problems posed before checking the
`solutions` branch.

## Setup Requirements

In order to complete the assignments, fellows will need to first complete the
following steps. Contact your facilitator if you run into issues.

1. Install `conda` (Anaconda or Miniconda is fine). Follow the steps for your operating system:
[Windows](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html),
[macOS](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html),
[Linux](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)
2. Clone this repository: `git clone https://github.com/AIS-UCLA/cv-fellowship.git`
3. Create a conda environment: `conda create --name ENV_NAME python=3.10 && conda activate ENV_NAME`
4. Verify installation: `python -V` should return something like `Python 3.10.12`
5. Install dependencies `pip install pytest jupyter keras tensorflow matplotlib scikit-learn`
6. Follow the [torch installation guide for your system](https://pytorch.org/get-started/locally/),
using `pip`. If you have a mac, choose default compute platform. If you are on 
Windows or Linux, choose CUDA 11.8 if you have an NVIDIA GPU, otherwise choose
CPU. (AMD users may attempt to use ROCm at their own risk.)
