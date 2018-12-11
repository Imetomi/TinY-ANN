/*
 * Description: this library contains functions that help you to create
 * artificial neural networks in C language. You can use this project to
 * implement neural networks on embedded systems (ex.: Arduino) or run
 * smaller networks on your computer.
 *
 * Made by Tamás Imets
 * Date: 18th of November, 2018
 * Version: 0.1
 * Github: https://github.com/Imetomi
 *
 * Note: SDL2 is needed for visualisations. If you don't want to use this
 * function remove every SDL related file and function.
 *
 */

#ifndef NEURAL_NETWORK_IN_C_PERCEPTRON_H
#define NEURAL_NETWORK_IN_C_PERCEPTRON_H


#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <SDL2/SDL.h> //Remove these if you don't want to use SDL
#include <SDL2/SDL2_gfxPrimitives.h> //Remove these if you don't want to use SDL


/* Structure to store dimensions for datasets, matrices, etc... */
typedef struct Dim {
    int h;
    int w;
} Dim;


/* Structure for a layer */
typedef struct Layer {
    Dim dim;
    float *in;
    float *out;
    float **weights;
    struct Layer *next, *prev;
} Layer;


/* Doubly linked list for a neural network */
typedef struct NeuralNet {
    Layer *input, *output;
} NeuralNet;


SDL_Event ev;

/* Functions in perceptron.c */
void end(); /* Terminates program */
float dist(float ax, float ay, float bx, float by); /* Returns the distance between two points */
float sigmoid(float x); /* Sigmoid activation function */
float sigmoid_der(float x); /* Derivative of sigmoid */
float sum(const float *v, int n); /* Sum of the elements of an array */
float dot_product(float *v, float *u, int n); /* Dot product of two arrays */
float rand_float(); /* Returns arandom float between 0 and 1 */
float *allocate_float_1d(int n); /* Dynamically allocating memory for an float type array */
float **allocate_float_2d(int n, int m); /* Dynamically allocating memorty for a 2d array */
void swap_float(float *a, float *b); /* Swap two given variables */
void free_float_1d(float *v); /* Free function for a 1d array */
void free_float_2d(float **v, int n);
void mini_max(float *v, int n, float *max, float *min); /* Looks for the min and max value in an array */
void fill_zero(float *v, int n); /* Fills an array with zeros */
void fill_one(float *v, int n); /* Fills an array with ones*/
void standard_scaler(float **v, Dim dim); /* Standardization - Feature scaling */
void minmax_scaler(float **v, Dim dim); /* Min max Feature Scaling */
/* Reads data from a CSV */
void read_csv(FILE *file, float **X_train, float **X_test, float **y_train, float **y_test, Dim train_dim, Dim test_dim);
void create_clusters(float **X, float **y, int n); /* Creates linearly separable datasets for training */
void create_circles(float **X, float **y, int n); /* Creates two circle datasets */
void create_spiral(float **X, float **y, int n); /* Creates an Archimedean spiral */
void create_chesstable(float **X, float **y, int n, float dist); /* Creates a chesstable pattern */
/* Splits training and testing data */
void split_train_test(float **X, float **y, float **X_train, float **X_test, float **y_train,
                      float **y_test, Dim dim, float ratio);
float *get_row(float **v, int h, int idx);


/* Functions in perceptron_plotter.c
 * Remove these if you don't want to use SDL
 * */
Uint32 timer(Uint32 ms, void *param); /* Timer for SDL */
void plot_init(SDL_Window **pwindow, SDL_Renderer **prenderer); /* Initialize SDL */
void plot_error_scaled(struct SDL_Renderer *renderer, float *J, int step, Uint32 color);
void plot_accuracy_scaled(struct SDL_Renderer *renderer, float *acc, int step, Uint32 color);
/* Uses SDL2 to visualize a 2D dataset */
void plot_clusters(struct SDL_Renderer *renderer, float **X, float **y, int output_dim);
void plot_trained_net(struct SDL_Renderer *renderer, NeuralNet *ann); /* Visualises trained net */


/* Functions in perceptron_libs.c */
NeuralNet *create_net(Dim in, Dim out); /* Creates a neural net with one hidden layer */
void add_hidden_layer(NeuralNet *ann, int layer_size); /* Inserts a hidden layer between the input and the second layer */
void print_net(NeuralNet *ann); /* Prints the weight matrices */
void free_net(NeuralNet *ann); /* Free allocated memory */
void feed_forward_net(NeuralNet *ann, float *X); /* Feeds forward information  */
/* Trains network */
void train_net(NeuralNet *ann, float **X, float **y, float *J, float* acc, Dim dim, int n_epoch);
/* Validates network */
void test_net(NeuralNet *ann, float **X, float **y, Dim dim);

#endif //NEURAL_NETWORK_IN_C_PERCEPTRON_H














