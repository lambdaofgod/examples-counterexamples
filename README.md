# examples-counterexamples

## Examples of some machine learning-related facts and libraries

### Requirements
To run examples using code from `src` you'll need to run `python setup.py develop`. 

Some examples also require `graphviz` installed (on Ubuntu you can do `apt-get install graphviz`) 

For loading data for word embeddings or coil20 dataset see `Makefile`.

## Notebooks

* [Kernel SVMs: fitting sine function, nonlinear classification with RBF kernel](
https://github.com/lambdaofgod/examples-counterexamples/blob/master/notebooks/Kernel%20SVMs.ipynb)

* [Transforming nonlinear data with Kernel PCA](
https://github.com/lambdaofgod/examples-counterexamples/blob/master/notebooks/Kernel%20PCA.ipynb)

* [(counterexample) if a dataset is linearly separable then it is linearly separable after applying PCA](https://github.com/lambdaofgod/examples-counterexamples/blob/master/notebooks/Separable%20data%20PCA%20nonseparable.ipynb)

* [Extractive Summarization](https://github.com/lambdaofgod/examples-counterexamples/blob/master/notebooks/text_mining/Extractive_Summarization.ipynb) in Python, includes one word-embedding based method.

* [Nonnegative Matrix Factorization (NMF) for topic modeling](https://github.com/lambdaofgod/examples-counterexamples/blob/master/notebooks/text_mining/NMF%20for%20topic%20modeling.ipynb) - very simple exaple on 20 Newsgroups 

### Deep learning 

* [Symmetric Deep Dream](https://github.com/lambdaofgod/examples-counterexamples/blob/master/notebooks/neural_nets/deepdream_symmetric.ipynb) - added rotation/mirroring invariance to original Deep Dream code

* [Variational Autoencoder (VAE) in Keras](https://colab.research.google.com/drive/1iQ4rX3Xr2YJpaMlO_98XldvRGCPZESRA) - Example of running generative model on very small dataset (Olivetti faces dataset - 400 examples). Adapted and refactored code from Keras project.

* [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) - [using Theano](https://github.com/lambdaofgod/examples-counterexamples/blob/master/notebooks/neural_nets/Logistic%20Regression%20with%20Theano.ipynb) [(code)](https://github.com/lambdaofgod/examples-counterexamples/blob/master/src/neural_nets/theano/logistic_regression.py)  

* [Basic MXNet - logistic regression & MLP](https://github.com/lambdaofgod/examples-counterexamples/blob/master/notebooks/neural_nets/MXNet%20basics.ipynb)

* [GANs in Gluon](https://colab.research.google.com/drive/1Esr2lPhx9yKPYyMvJIGWKt68YjEJQMg7) - using Gluon's Hybrid Blocks which utilize dynamic graphs by default, but can be compiled to static ones

* [Visualizing Convolutional Neural Networks in Keras with Quiver](https://github.com/lambdaofgod/examples-counterexamples/blob/master/notebooks/neural_nets/Keras%20CNN%20visualization%20with%20Quiver.ipynb)

### Other

* [Minimal CoreNLP Scala notebook](https://github.com/lambdaofgod/examples-counterexamples/blob/master/notebooks/CoreNLP.ipynb)
