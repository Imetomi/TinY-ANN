![alt text](https://github.com/Imetomi/TinY-ANN/blob/master/img/tinyann.png)

# Artificial Neural Network Library in C

TinY ANN is a simple library to create neural networks in C for smaller data science projects. You can use this library to implement neural networks on Arduinos (maybe other embedded systems too) or to make complete data science projects in C. The repository and the library contains the most important data managing and feature scaling algorithms too. The prewritten functions do all the hard job so it feels like creating an ANN in Python.

## Setting up TinY ANN

All you have to do is to add `perceptron.h`, `perceptron.c`, `perceptron_libs.c`, and `perceptron_plotter.c` to your project then include the header file. 

**NOTE: `perceptron_plotter.c` uses SDL2 library to make graphical visualizations. If don't want to use the graphical tools then simply remove every related function and file. I have marked these in the code, feel free to modify it.**

## Example Codes

There are 6 example codes in this repository. These examples show you how to use the library, how to create neural networks, and how to train them. There are four graphical examples and two data science projects in which I solve two very pupular machine learning problems.

### Graphical Examples

#### Grahical Visualization using SDL2

You'll need to have SDL2 in order to try out these example codes. The library creates you a canvas on which you can draw whatever you want. There are some premade fuctions that help you to create good looking visualizations on the accuracy or error loss values. The library also contains functions that plot the training dataset on the canvas and visualize what the network learned.

There are some random dataset generating algorithms implemented in the library. You can use these for testing.

- `example_circle.c`	Creates two circle datasets and learns to classify the inner and the outer circle.
- `example_linear.c`	Creates two linearly separable datasets and learns to classify them.
- `example_check.c`		Creates a check table pattern and learns to classify the elements in it.
- `example_spiral.c`	Creates an Archimedean spiral and learns to classify the two rolls. 

![alt text](https://github.com/Imetomi/TinY-ANN/blob/master/img/plot.png)
	
#### Plot Gallery

On the image below you can see some of the most beautiful plots. The dots mark the training dataset and the lighter colors show how the neural net learned to classify this two dimensional dataset.

![alt text](https://github.com/Imetomi/TinY-ANN/blob/master/img/plot_gallery.png)

### Tackling Data Science Projects

#### Kaggle Titanic Dataset

This example shows how to use the `read_csv()` and `standard_scaler()` functions. Using a 3 layer (17-4-1) network I was able to get about 92% accuracy. To try out this example compile `example_titanic.c`. Note: the input data was cleaned in Jupyter. 

#### Red Wine Dataset

The `example_wine.c` example code tackles another very popular machine learning problem. In this case the neural network learns to classify good and bad red wines based on what chemicals they contain.

## Usage

It is relatively easy to use this library in C, just don't forget to free up the allocated memory. The hard job is done by the prewritten functions, you only have to deal with creating and training the network.


```C
#include "perceptron.h"

int main() {
    int n_epoch = 200; //number of learning epochs
    float eta = 0.09; //learnig rate
    Dim train_dim = {500, 8}; //training data dimension
    Dim test_dim = {100, 8}; //testing_data dimension

    /* defining dataset */
    float **X_train = allocate_float_2d(train_dim.h, train_dim.w);
    float **X_test = allocate_float_2d(train_dim.h, train_dim.w);
    float **y_train = allocate_float_2d(train_dim.h, 1);
    float **y_test = allocate_float_2d(test_dim.h, 1);
    /* for now we assume that you also fill these up with some useful information. */

    Dim in = {8, 5}; //first layer: 8 input neurons and 5 neurons in the hidden layer
    Dim out = {5, 1}; //second layer: 5 neurons in the hidden layer and 1 output neuron

    /* creating the neural net */
    NeuralNet *ann = create_net(in, out); //initialize a 8-5-1 neural network
    add_hidden_layer(ann, 5); //add another hidden layer
    /* the final network looks like this: 8-5-5-1 */

    /* training the neural network */
    train_net(ann, X_train, y_train, J, acc, train_dim, eta, n_epoch);

    /* testing the neural network */
    test_net(ann, X_test, y_test, test_dim);

    /* free up allocated memory */
    free_net(ann);
    free_float_2d(X_train, train_dim.h);
    free_float_2d(X_test, test_dim.h);
    free_float_2d(y_train, train_dim.h);
    free_float_2d(y_test, test_dim.h);

    return 0;
}
```