/*
 * Description: Example code for simple linear classification using
 * a neural network.
 *
 * Made by Tamás Imets
 * Date: 18th of November, 2018
 * Version: 0.1
 * Github: https://github.com/Imetomi
 *
 */

#include "perceptron.h"
#include "debugmalloc.h"


int main(int argc, char **argv) {
    srand(time(NULL));
    SDL_Renderer *renderer;
    SDL_Window *window;

    Dim dim = {100, 3};
    float **X, **y, *J, *acc, eta = 0.05;
    int n_epoch = 200;
    clock_t start, end;

    /* Defining first and second weight matrix dimensions */
    Dim in = {3, 5};
    Dim out = {5, 1};


    /* Creating a dataset */
    X = allocate_float_2d(dim.h, dim.w);
    y = allocate_float_2d(dim.h, out.w);
    J = allocate_float_1d(n_epoch);
    acc = allocate_float_1d(n_epoch);
    create_clusters(X, y, dim.h);


    /* Creating the neural network */
    NeuralNet *ann;
    ann = create_net(in, out);
    start = clock();
    train_net(ann, X, y, J, acc, dim, eta, n_epoch);
    end = clock();
    float cpu_time_used = (float) (end - start) / CLOCKS_PER_SEC;


    /* Visualizing data */
    plot_init(&window, &renderer);
    plot_clusters(renderer, X, y, dim.h);
    plot_error_scaled(renderer, J, n_epoch - 1, 0x000000FF);
    plot_accuracy_scaled(renderer, acc, n_epoch - 1, 0x000000FF);
    char err[30], accuracy[30], tmp[30];
    sprintf(tmp, "Time: %.2fs", cpu_time_used);
    sprintf(err, "Train Loss:  %.3f", J[n_epoch - 1]);
    sprintf(accuracy, "Accuracy:  %.3f", acc[n_epoch - 1]);
    stringRGBA(renderer, 640, 40, err, 190, 0, 140, 255);
    stringRGBA(renderer, 640, 340, accuracy, 255, 194, 0, 255);
    stringRGBA(renderer, 1070, 40, tmp, 0, 0, 0, 255);
    plot_trained_net(renderer, ann);
    SDL_RenderPresent(renderer);

    while (SDL_WaitEvent(&ev) && ev.type != SDL_QUIT) {
        if (ev.type == SDL_KEYUP && ev.key.keysym.sym == SDLK_ESCAPE) {
            SDL_Quit();
        }
    }


    /* Free up allocated memory */
    free_net(ann);
    free_float_1d(acc);
    free_float_2d(y, dim.h);
    free_float_1d(J);
    free_float_2d(X, dim.h);


    /* Terminate program */
    debugmalloc_dump();
    return 0;
}
