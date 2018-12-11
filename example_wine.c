/*
 * Description: Another machine learning project ->
 * Here we are going to teach an ANN to classify wines based
 * on what chemicals do they contain.
 *
 * Attribute information:
 * There are 1599 instances with 11 attributes each
 * For more information, read [Cortez et al., 2009].
 *
 *  Input variables (based on physicochemical tests):
 *  1 - fixed acidity (tartaric acid - g / dm^3)
 *  2 - volatile acidity (acetic acid - g / dm^3)
 *  3 - citric acid (g / dm^3)
 *  4 - residual sugar (g / dm^3)
 *  5 - chlorides (sodium chloride - g / dm^3
 *  6 - free sulfur dioxide (mg / dm^3)
 *  7 - total sulfur dioxide (mg / dm^3)
 *  8 - density (g / cm^3)
 *  9 - pH
 *  10 - sulphates (potassium sulphate - g / dm3)
 *  11 - alcohol (% by volume)
 *
 *  Output variable (based on sensory data):
 *  12 - quality (score between 0 and 10)
 *
 *
 * Made by Tamás Imets
 * Date: 18th of November, 2018
 * Version: 0.1
 * Github: https://github.com/Imetomi
 *
 */

#include "perceptron.h"

int main(int argc, char **argv) {
    /* Creating datasets and reading the data */
    Dim train_dim = {1280, 11};
    Dim test_dim = {318, 11};
    float **X_train, **X_test, **y_train, **y_test;

    /* Reading in the training dataset !!!CHANGE PATH!!! */
    /* The data was already cleaned and scaled in python */
    FILE* wine_data = fopen("C:\\YOUR_PATH_HERE\\data\\wine_data.csv", "r");
    /* READING FROM STANDARD INPUT */

    /* Reading the data */
    X_train = allocate_float_2d(train_dim.h, train_dim.w);
    X_test = allocate_float_2d(test_dim.h, test_dim.w);
    y_train = allocate_float_2d(train_dim.h, 1);
    y_test = allocate_float_2d(test_dim.h, 1);
    read_csv(wine_data, X_train, X_test, y_train, y_test, train_dim, test_dim);

    /* Good wines are rated above 5 :) */
    for (int i = 0; i < train_dim.h; ++i)
        y_train[i][0] = y_train[i][0] >= 7;

    for (int i = 0; i < test_dim.h; ++i)
        y_test[i][0] = y_test[i][0] >= 7;


    /* Scaling the data */
    standard_scaler(X_train, train_dim);
    standard_scaler(X_test, test_dim);


    /* Creating neural network */
    int n_epoch = 251;
    float *J, *acc;
    J = allocate_float_1d(n_epoch);
    acc = allocate_float_1d(n_epoch);
    Dim in = {11, 6};
    Dim out = {6, 1};
    NeuralNet *ann = create_net(in, out);
    feed_forward_net(ann, X_train[0]);

    /* Training network on the training samples */
    train_net(ann, X_train, y_train, J, acc, train_dim, n_epoch);


    /* Testing accuracy on the testing samples */
    test_net(ann, X_test, y_test, test_dim);


    /* Free up allocated memory */
    free_float_1d(J);
    free_float_1d(acc);
    free_float_2d(X_train, train_dim.h);
    free_float_2d(X_test, test_dim.h);
    free_float_2d(y_train, train_dim.h);
    free_float_2d(y_test, test_dim.h);
    free_net(ann);

    return 0;
}