/*
 * Machine Learning from a Disaster
 *
 * Description: A complete data science project in C
 * This program tackles the famous Titanic Machine Learning
 * Challange. We will teach a neural network to predict if
 * someone would survive the journey on the Titanic or not.
 *
 * Made by Tamás Imets
 * Date: 18th of November, 2018
 * Version: 0.1
 * Github: https://github.com/Imetomi
 *
 * Note: The dataset was half cleaned in Python already.
 * I made label binarization on the dataset.
 * There are 17 attributes per instance. The last element is
 * actually the label of the instance (1 = survived, 0 = didn't survive).
 * There are 891 instances in the whole dataset. I will use the first 700
 * as training samples. The remaining 191 will be used as testing data.
 *
 */

#include "perceptron.h"

int main(int argc, char **argv) {
    /* Creating datasets and reading the data */
    Dim train_dim = {700, 17};
    Dim test_dim = {190, 17};
    float **X_train, **X_test, **y_train, **y_test;
    X_train = allocate_float_2d(train_dim.h, train_dim.w);
    X_test = allocate_float_2d(test_dim.h, test_dim.w);
    y_train = allocate_float_2d(train_dim.h, 1);
    y_test = allocate_float_2d(test_dim.h, 1);


    /* Reading in the training dataset !!!CHANGE PATH!!! */
    /* The data was already cleaned and scaled in python */
    FILE* titanic_data = fopen("YOUR_PATH\\data\\titanic_data.csv", "r");
    read_csv(titanic_data, X_train, X_test, y_train, y_test, train_dim, test_dim);

    /* Scaling the data */
    standard_scaler(X_train, train_dim);
    standard_scaler(X_test, test_dim);


    /* Creating neural network */
    int n_epoch = 401;
    float eta = 0.09, *J, *acc;
    J = allocate_float_1d(n_epoch);
    acc = allocate_float_1d(n_epoch);
    Dim in = {17, 4};
    Dim out = {4, 1};
    NeuralNet *ann = create_net(in, out);


    /* Training network on the training samples */
    train_net(ann, X_train, y_train, J, acc, train_dim, eta, n_epoch);

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
