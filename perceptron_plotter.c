/*
 * This file contains every function that is needed to create
 * great looking plots for a 2D dataset. Take a look at the examples
 * in order to understand how to use these functions.
 *
 * Made by Tamás Imets
 * Date: 18th of November, 2018
 * Version: 0.1.1
 * Github: https://github.com/Imetomi
 *
 * Important Note: If you don't want to use SDL at all then you can remove this file.
 * In this case remove all the SDL related functions from the perceptron.h file.
 *
 */

#include "perceptron.h"
#include "debugmalloc.h"

const float Height = 600.0, Width = 1200.0, Margin = 30;

/* Timer for SDL */
Uint32 timer(Uint32 ms, void *param) {
    SDL_Event ev;
    ev.type = SDL_USEREVENT;
    SDL_PushEvent(&ev);
    return ms;
}

/* Initializes SDL Window */
void plot_init(SDL_Window **pwindow, SDL_Renderer **prenderer) {
    char title[] = "Neural Network Visualizer";
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
        SDL_Log("Could not launch SDL: %s", SDL_GetError());
        exit(1);
    }

    SDL_Window *window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int) Width, (int) Height, 0);
    if (window == NULL) {
        SDL_Log("Window cannot be created: %s", SDL_GetError());
        exit(1);
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == NULL) {
        SDL_Log("Renderer error: %s", SDL_GetError());
        exit(1);
    }

    SDL_RenderClear(renderer);

    *pwindow = window;
    *prenderer = renderer;
}


/* Uses SDL2 to visualize a 2D dataset */
void plot_clusters(struct SDL_Renderer *renderer, float **X, float **y, int output_dim) {
    boxRGBA(renderer, 0, 0, (Sint16) Width, (Sint16) Height, 255, 255, 255, 255);
    int R = 4;
    for (int i = 0; i < output_dim; ++i) {
        Sint16 poz_x = (Sint16) (Width * X[i][1] / 2);
        Sint16 poz_y = (Sint16) (Height * X[i][2]);

        if (poz_x > Margin + 3 && poz_x < Width / 2 - Margin - 3 &&
            poz_y > Margin + 3 && poz_y < Height - Margin - 3) {
            if (y[i][0] > 0.5) {
                filledCircleRGBA(renderer, poz_x, poz_y, R, 69, 14, 97, 255);
            } else {
                filledCircleRGBA(renderer, poz_x, poz_y, R, 252, 200, 0, 255);
            }
        }
    }

    rectangleRGBA(renderer, (Sint16) Margin, (Sint16) Margin, (Sint16) (Width / 2 - Margin), (Sint16) (Height - Margin), 0, 0, 0, 255);
    rectangleRGBA(renderer, (Sint16) (Width / 2 + Margin), (Sint16) (Height / 2 - Margin), (Sint16) (Width - Margin), (Sint16) (Margin), 0, 0, 0, 255);
    rectangleRGBA(renderer, (Sint16) (Width / 2 + Margin), (Sint16) (Height / 2 + Margin), (Sint16) (Width - Margin), (Sint16) (Height - Margin), 0, 0, 0, 255);
}



/* Able to visualize every function generated class */ /*
void plot_trained_net(struct SDL_Renderer *renderer, NeuralNet *ann) {
    float step_i = (float) 1 / (Width / 2);
    float step_j = (float) 1 / Height;
    float z = 0.5;

    // Float loop corrected with Machine Epsilon
    for (float i = Margin / (Width / 2); i < 1 - Margin / (Width / 2) + MACHINE_EPSILON; i += step_i) {
        for (float j = Margin / Height; j < 1 - Margin / (Width / 2) + MACHINE_EPSILON; j += step_j) {
            float pixel[8] = {1, i, j, (float) sin(i * 10), (float) sin(j * 10), i * j, i * i, j * j};
            feed_forward_net(ann, pixel);
            float res = ann->output->out[0];
            if (res >= z) {
                pixelRGBA(renderer, (Sint16) ((Width / 2) * pixel[1]), (Sint16) (Height * pixel[2]),
                          130, 0, 120, (Uint8) ((res - 0.5) * 255));
            } else {
                pixelRGBA(renderer, (Sint16) ((Width / 2) * pixel[1]), (Sint16) (Height * pixel[2]),
                          255, 194, 0, (Uint8) ((0.5 - res) * 255));
            }
        }
    }
}
 */

void plot_trained_net(struct SDL_Renderer *renderer, NeuralNet *ann) {
    float z = 0.5;
    float size = 540.0;

    // Float loop corrected with Machine Epsilon
    float i, j;
    for (int x = (int) Margin - 1; x < Height - Margin ; ++x) {
        for (int y = (int) Margin - 1; y < Height - Margin; ++y) {
            i =((float) x - Margin)/ size;
            j = ((float) y - Margin) / size;

            float pixel[8] = {1, i, j, (float) sin(i * 10), (float) sin(j * 10), i * j, i * i, j * j};
            feed_forward_net(ann, pixel);
            float res = ann->output->out[0];
            if (res >= z) {
                pixelRGBA(renderer, (Sint16) y, (Sint16) x,
                          130, 0, 120, (Uint8) ((res - 0.5) * 255));
            } else {
                pixelRGBA(renderer, (Sint16) y, (Sint16) x,
                          255, 194, 0, (Uint8) ((0.5 - res) * 255));
            }
        }
    }
}


/* Plots error graph on SDL window */
void plot_error_scaled(struct SDL_Renderer *renderer, float *J, int step, Uint32 color) {
    Sint16 def_poz_x = Width / 2 + Margin;
    Sint16 def_poz_y = Margin + 20;

    float min, max;
    mini_max(J, step, &max, &min);

    for (int i = 0; i < step; ++i) {
        float scaled = (max - J[i]) * ((Height / 2 - 3 * Margin) / max);
        Sint16 poz_x = Width / 2 + Margin + i * (Width / 2 - 2 * Margin) / (step - 1);
        Sint16 poz_y = scaled + Margin + 20; // +20 to look better

        lineColor(renderer, def_poz_x, def_poz_y, poz_x, poz_y, color);

        def_poz_x = poz_x;
        def_poz_y = poz_y;
    }
}


/* Plots accuracy graph on SDL window */
void plot_accuracy_scaled(struct SDL_Renderer *renderer, float *acc, int step, Uint32 color) {
    Sint16 def_poz_x = Width / 2 + Margin;
    Sint16 def_poz_y = 520;

    float min, max;
    mini_max(acc, step, &max, &min);

    for (int i = 0; i < step; ++i) {
        float scaled = (max - acc[i]) * ((Height / 2 - 3 * Margin) / max);
        Sint16 poz_x = Width / 2 + Margin + i * (Width / 2 - 2 * Margin) / (step - 1);
        Sint16 poz_y = scaled + Height / 2 + Margin + 40; // +20 to look better

        lineColor(renderer, def_poz_x, def_poz_y, poz_x, poz_y, color);

        def_poz_x = poz_x;
        def_poz_y = poz_y;
    }
}
