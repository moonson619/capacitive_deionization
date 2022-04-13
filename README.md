# Capacitive Deionization (also known as CDI)
This Gitbub repository contains python-based deep learning models for capacitive deionization process.

For the detailed experiment conditions and information about the model, please see the research paper below.
https://www.sciencedirect.com/science/article/pii/S0011916421003040

*Another paper is Under Review.

# Environments
The codes have been tested under the environment as follows:
- python 3.7
- Anaconda 3
- Visual Studio Code (version 1.66)
- Computer
    * CPU: 11th Gen Intel(R) Core(TM) i9-11900K
    * GPU: NVIDIA GeForce RTX 3090 24 GB (Note that the TensorFlow V 1.15, which is not compatible with this GPU, was used)
    * RAM: 128 GB
    * OS: Windows 10

# Installation
Python can be installed through an anaconda command as follows:

conda create -n name python=3.7

Install required packages using "pip install" and enclosed "requirements.txt" file as follows:

pip install -r requirements.txt

After installation, you need to check that all packages are installed properly as described in the "requirements.txt" file.

# How to use
For your own use:
1. Use "HyperOpt" code for Hyperparameter optimization
2. Use "RunModel" code for prediction (test)

Import pre-trained model:
1. Import pre-trained models under "trained_model" folder
2. Run the model using a code under "codes" folder
