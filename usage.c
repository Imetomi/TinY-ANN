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