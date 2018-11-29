/*
 * This file contains functions that you can use directly to
 * create and train a neural network. Everything is allocated dynamically
 * so you will have to free up memory at the end of your
 * program. Feel free to modify and edit the code.
 *
 * Made by Tamás Imets
 * Date: 18th of November, 2018
 * Version: 0.1v
 * Github: https://github.com/Imetomi
 *
 */


#include "perceptron.h"
#include "debugmalloc.h"


/* Prints the weight matrices in a neural net */
void print_net(NeuralNet *ann) {
    Layer *iter;
    for (iter = ann->input; iter != NULL; iter = iter->next) {
        printf("\nLayer size = %d: ", iter->dim.h);
        for (int i = 0; i < iter->dim.w; ++i) {
            printf("%f ", iter->out[i]);
        }
        printf("\n");

        for (int i = 0; i < iter->dim.h; ++i) {
            for (int j = 0; j < iter->dim.w; ++j) {
                printf("%f ", iter->weights[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


/* Free function for the whole neural net */
void free_net(NeuralNet *ann) {
    Layer *iter = ann->input;
    while (iter != NULL) {
        Layer *next = iter->next;
        free(iter->in);
        free(iter->out);
        free_float_2d(iter->weights, iter->dim.h);
        free(iter);
        iter = next;
    }

    free(ann);
}


/* Initializes a random weight matrix */
void init_weight_matrix(float **w, Dim dim) {
    for (int i = 0; i < dim.h; i++) {
        for (int j = 0; j < dim.w; ++j) {
            w[i][j] = rand_float() - (float) 0.5;
        }
    }
}


/* Allocates memory and creates a neural net */
NeuralNet *create_net(Dim in, Dim out) {
    NeuralNet *ann = (NeuralNet*) malloc(sizeof(NeuralNet));
    ann->input = (Layer*) malloc(sizeof(Layer));
    ann->output = (Layer*) malloc(sizeof(Layer));
    ann->input->prev = NULL;
    ann->input->next = ann->output;
    ann->output->prev = ann->input;
    ann->output->next = NULL;

    ann->input->dim.h = in.h;
    ann->input->dim.w = in.w;
    ann->output->dim.h = out.h;
    ann->output->dim.w = out.w;

    ann->input->weights = allocate_float_2d(ann->input->dim.h, ann->input->dim.w);
    ann->output->weights = allocate_float_2d(ann->output->dim.h, ann->output->dim.w);
    ann->input->in = allocate_float_1d(ann->input->dim.w + ann->input->dim.h);
    ann->input->out = allocate_float_1d(ann->input->dim.w + ann->input->dim.h);
    ann->output->in = allocate_float_1d(ann->output->dim.h);
    ann->output->out = allocate_float_1d(ann->output->dim.h);

    init_weight_matrix(ann->input->weights, ann->input->dim);
    init_weight_matrix(ann->output->weights, ann->output->dim);

    fill_zero(ann->input->in, ann->input->dim.h);
    fill_zero(ann->input->out, ann->input->dim.h);
    fill_zero(ann->output->in, ann->output->dim.h);
    fill_zero(ann->output->out, ann->output->dim.h);

    return ann;
}


/* Adds a new layer to the neural nerwork */ /*
void add_hidden_layer(NeuralNet *ann, int layer_size) {
    Layer *new = (Layer*) malloc(sizeof(Layer));
    ann->input->dim.w = layer_size;
    new->dim.h = layer_size;
    new->dim.w = ann->input->next->dim.h;
    new->in = allocate_float_1d(new->dim.w);
    new->out = allocate_float_1d(new->dim.w);
    fill_zero(new->in, new->dim.w);
    fill_zero(new->out, new->dim.w);

    free_float_2d(ann->input->weights, ann->input->dim.h);
    free_float_2d(ann->input->next->weights, ann->input->next->dim.h);
    free_float_1d(ann->input->in);
    free_float_1d(ann->input->out);

    new->weights = allocate_float_2d(new->dim.h, new->dim.w);
    ann->input->weights = allocate_float_2d(ann->input->dim.h, ann->input->dim.w);
    ann->input->next->weights = allocate_float_2d(ann->input->next->dim.h, ann->input->next->dim.w);
    ann->input->in = allocate_float_1d(ann->input->dim.w);
    ann->input->out = allocate_float_1d(ann->input->dim.w);

    init_weight_matrix(ann->input->weights, ann->input->dim);
    init_weight_matrix(ann->input->next->weights, ann->input->next->dim);
    init_weight_matrix(new->weights, new->dim);
    fill_zero(ann->input->in, ann->input->dim.w);
    fill_zero(ann->input->out, ann->input->dim.w);

    new->next = ann->input->next;
    new->prev = ann->input;
    ann->input->next = new;
    ann->input->next->prev = new;
}
 */


/* Feeds forward data in the neural network*/
void feed_forward_net(NeuralNet *ann, float *X) {
    for (int i = 0; i < ann->input->dim.w; ++i) {
        float *r = get_row(ann->input->weights, ann->input->dim.h, i);
        ann->input->in[i] = dot_product(X, r, ann->input->dim.h);
        ann->input->out[i] = sigmoid(ann->input->in[i]);
        free(r);
    }

    Layer *iter;
    for (iter = ann->input->next; iter != NULL; iter=iter->next) {
        for (int i = 0; i < iter->dim.w; ++i) {
            float *r = get_row(iter->weights, iter->dim.h, i);
            iter->in[i] = dot_product(iter->prev->out, r, iter->dim.h);
            iter->out[i] = sigmoid(iter->in[i]);
            free(r);
        }
    }
}


/* Trains the neural network  */
void train_net(NeuralNet *ann, float **X, float **y, float *J, float *acc, Dim dim, int n_epoch) {
    float *delta_second_layer = allocate_float_1d(ann->output->dim.h);
    clock_t start, end;
    start = clock();

    for (int step = 0; step < n_epoch; ++step) {
        float sum_err = 0;
        int correct = 0;

        for (int i = 0; i < dim.h; ++i) {
            feed_forward_net(ann, X[i]);

            if ((int) (ann->output->out[0] + 0.5) == (int) y[i][0])
                correct++;

            float error_last_layer = y[i][0] - ann->output->out[0];
            float delta_last_layer = error_last_layer * sigmoid_der(ann->output->out[0]);

            for (int j = 0; j < ann->output->dim.h; ++j) {
                delta_second_layer[j] = ann->output->weights[j][0] * delta_last_layer * sigmoid_der(ann->output->prev->out[j]);
            }

            for (int j = 0; j < ann->output->dim.h; ++j) {
                for (int k = 0; k < ann->output->dim.w; ++k) {
                    ann->output->weights[j][k] += delta_last_layer * ann->input->out[j];
                }
            }

            for (int j = 0; j < ann->input->dim.h; ++j) {
                for (int k = 0; k < ann->input->dim.w; ++k) {
                    ann->input->weights[j][k] += X[i][j] * delta_second_layer[k];
                }
            }

            sum_err += error_last_layer * error_last_layer * (float) 0.5;
        }

        J[step] = sum_err;
        acc[step] = (float) correct / (float) dim.h;

        if (step % 50 == 0)
            printf("Epoch: %d   Error: %0.3f   Accuracy: %0.3f\n", step, J[step], acc[step]);
    }

    free_float_1d(delta_second_layer);
    end = clock();
    float training_time = (float) (end-start) / CLOCKS_PER_SEC;
    printf("Training took: %0.3f sec\n", training_time);
}


/* Testing accuracy on the given neural network  */
void test_net(NeuralNet *ann, float **X, float **y, Dim dim) {
    int correct = 0;
    float rmse = 0;
    float mae = 0;

    for (int i = 0; i < dim.h - 1; ++i) {
        feed_forward_net(ann, X[i]);
        if ((int) (ann->output->out[0] + 0.5) == (int) y[i][0])
            correct++;

        rmse += (y[i][0] - ann->output->out[0]) * (y[i][0] - ann->output->out[0]);
        mae += fabs((double) (y[i][0] - ann->output->out[0]));
    }

    rmse = (float) sqrt((double) (rmse / (float) dim.h));
    mae = mae / (float) dim.h;
    printf("\nTest Accuracy: %f   Correct: %d   Misclassified: %d\n",
            (float) correct / (float) dim.h,
            correct, dim.h - correct);
    printf("Root Mean Squared Error: %f\n", rmse);
    printf("Mean Absolute Error: %f\n", mae);
}

