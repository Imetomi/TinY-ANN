#include "perceptron.h"

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


/* Adds a new layer to the neural nerwork */
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
void train_net(NeuralNet *ann, float **X, float **y, float *J, float *acc, Dim dim, float eta, int n_epoch) {
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



/* Able to visualize every function generated class */
void plot_trained_net(struct SDL_Renderer *renderer, NeuralNet *ann) {
    float step_i = (float) 1 / (Width / 2);
    float step_j = (float) 1 / Height;
    float z = 0.5;

    for (float i = Margin / (Width / 2); i <= 1 - Margin / (Width / 2); i += step_i) {
        for (float j = Margin / Height; j <= 1 - Margin / (Width / 2); j += step_j) {
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

void end() {
    printf("Sorry bruh, code stopped... \n");
    exit(1);
}


/* Returns the distance between two points */
float dist(float ax, float ay, float bx, float by) {
    return (float) sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}


/* Sigmioid activation function */
float sigmoid(float x) {
    return (float) (1.0 / (1.0 + exp((double) -(x - (float) 0.5))));
}


/* Derivative of sigmoid */
float sigmoid_der(float x) {
    return x * (1 - x);
}


/* Sum of the elements of an array */
float sum(const float *v, int n) {
    float s = 0.0;
    for (int i = 0; i < n; ++i)
        s += v[i];
    return s;
}


/* Dot product of two arrays */
float dot_product(float *v, float *u, int n) {
    float result = 0.0;
    for (int i = 0; i < n; ++i)
        result += v[i] * u[i];
    return result;
}


/* Swap two given variables */
void swap_float(float *a, float *b) {
    float tmp = *a;
    *a = *b;
    *b = tmp;
}


/* Shuffle the elements of a float array */
void shuffle(float *v, int n) {
    for (int i = 0; i < n - 2; ++i) {
        int j = rand() % (n - i) + i;
        swap_float(&v[i], &v[j]);
    }
}


/* Dynamically allocating memory for an float type array */
float *allocate_float_1d(int n) {
    float *v = (float*) malloc(sizeof(float) * n);
    return v;
}


/* Dynamically allocating memorty for a 2d array */
float **allocate_float_2d(int n, int m) {
    float **X;
    X = (float**) malloc(sizeof(float*) * n);
    for (int i = 0; i < n; ++i)
        X[i] = (float*) malloc(m * sizeof(float));
    return X;
}


/* Free function for a 1d array */
void free_float_1d(float *v) {
    free(v);
}


/* Gets the ith row from the transpose of a matrix */
float *get_row(float **v, int h, int idx) {
    float *t = (float*) malloc(sizeof(float) * h);
    for (int i = 0; i < h; ++i) {
        t[i] = v[i][idx];
    }

    return t;
}


/* Looks for the min and max elements */
void mini_max(float *v, int n, float *max, float *min) {
    *max = v[0];
    *min = v[0];

    for (int i = 1; i < n; ++i) {
        if (v[i] > *max)
            *max = v[i];
        if (v[i] < *min)
            *min = v[i];
    }
}


/* Free function for a 2d array */
void free_float_2d(float **v, int n) {
    for (int i = 0; i < n; ++i) {
        free(v[i]);
    }
    free(v);
}


/* Fills an array with zeros */
void fill_zero(float *v, int n) {
    for (int i = 0; i < n; ++i)
        v[i] = 0.0;
}


/* Fills an array with ones*/
void fill_one(float *v, int n) {
    for (int i = 0; i < n; ++i)
        v[i] = 1.0;
}


/* creates arandom float between 0 and 1 */
float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}


/* Standardization - Feature Scaling */
void standard_scaler(float **v, Dim dim) {
    for (int j = 0; j < dim.w; ++j) {
        float mean = 0;
        for (int i = 0; i < dim.h; ++i)
            mean += v[i][j];
        mean /= (float) dim.h;

        float std_dev = 0;
        for (int i = 0; i < dim.h; ++i)
            std_dev += (v[i][j] - mean) * (v[i][j] - mean);
        std_dev = (float) sqrt((double) (std_dev / (float) dim.h));

        if (std_dev != 0)
            for (int i = 0; i < dim.h; ++i)
                v[i][j] = (v[i][j] - mean) / std_dev;
    }
}


/* Min-Max Feature Scaling */
void minmax_scaler(float **v, Dim dim) {
    for (int j = 0; j < dim.w; ++j) {
        float min = v[0][j];
        float max = v[0][j];

        for (int i = 0; i < dim.h; ++i) {
            if (v[i][j] < min)
                min = v[i][j];
            if (v[i][j] > max)
                max = v[i][j];
        }

        float diff = max - min;

        if (diff != 0)
            for (int i = 0; i < dim.h; ++i)
                v[i][j] = (v[i][j] - min) / diff;
    }
}

/* CSV Reader especially for this example */
void read_csv(FILE *file, float **X_train, float **X_test, float **y_train, float **y_test,
              Dim train_dim, Dim test_dim) {
    char line[200 + 1];
    int cnt = 0;
    while ((fgets(line, 1000, file) != NULL) && (cnt < train_dim.h)) {
        int idx;
        char *p;
        for (p = strtok(line, ","), idx = -1; p && *p && idx < train_dim.w; p = strtok(NULL, ","), ++idx) {
            if (idx >= 0) { //the first column is not needed
                X_train[cnt][idx] = (float) atof(p);
            }
        }

        y_train[cnt][0] = (float) atof(p); //the last 'p' pointer contains the label!
        ++cnt;
    }

    /* Reading in the testing datasets */
    cnt = 0;
    while ((fgets(line, 1000, file) != NULL) && (cnt < test_dim.h)) {
        int idx;
        char *p;
        for (p = strtok(line, ","), idx = -1; p && *p && idx < test_dim.w; p = strtok(NULL, ","), ++idx) {
            if (idx >= 0) { //the first column is not needed
                X_test[cnt][idx] = (float) atof(p);
            }
        }
        y_test[cnt][0] = (float) atof(p); //the last 'p' pointer contains the label!
        ++cnt;
    }
}

/* Creates linearly separable datasets for training */
void create_clusters(float **X, float **y, int n) {
    float A[2] = {rand_float(), rand_float()};
    float B[2] = {rand_float(), rand_float()};
    while (dist(A[0], A[1], B[0], B[1]) < 0.9) {
        A[0] = rand_float(); A[1] = rand_float();
        B[0] = rand_float(); B[1] = rand_float();
    }


    float size = rand_float() * (float) 1.1 - rand_float();
    while (size <= 0.3 || size >= 0.35) size = rand_float() * 1.1 - rand_float();

    int ok = 0;
    while (ok < n) {
        float a = rand_float();
        float b = rand_float();

        float dist_a = dist(a, b, A[0], A[1]);
        float dist_b = dist(a, b, B[0], A[1]);

        if (dist_a < dist_b) {
            if (dist_a < size) {
                y[ok][0] = 1;
                X[ok][0] = 1;
                X[ok][1] = a;
                X[ok][2] = b;
                ++ok;
            }
        } else {
            if (dist_b < size) {
                y[ok][0] = 0;
                X[ok][0] = 1;
                X[ok][1] = a;
                X[ok][2] = b;
                ++ok;
            }
        }
    }
}

void create_circles(float **X, float **y, int n) {
    int class = rand() % 2;
    int ok = 0;
    while (ok < n) {
        float a = rand_float();
        float b = rand_float();

        if (dist(a, b, 0.5, 0.5) < 0.4) {
            if (dist(a, b, 0.5, 0.5) < 0.15) {
                X[ok][0] = 1;
                X[ok][1] = a;
                X[ok][2] = b;
                y[ok][0] = class == 0;
                ++ok;
            } else if ((dist(a, b, 0.5, 0.5) > 0.25) ) {
                X[ok][0] = 1;
                X[ok][1] = a;
                X[ok][2] = b;
                y[ok][0] = class == 1;
                ++ok;
            }
        }
    }
}


/* Creates Archimede's spiral */
void create_spiral(float **X, float **y, int n) {
    float a = 0, b = 0.4;
    for (int i = 0; i < n; ++i) {
        if (rand() % 2 == 0) {
            float t = (float) i / ((float) n);
            X[i][0] = 1;
            X[i][1] = (float) 0.5 + (a + b * t) * (float) cos((double) t * 10);
            X[i][2] = (float) 0.5 + (a + b * t) * (float) sin((double) t * 10);
            X[i][3] = (float) sin((double) X[i][1] * 10);
            X[i][4] = (float) sin((double) X[i][2] * 10);
            X[i][5] = X[i][2] * X[i][1];
            X[i][6] = X[i][1] * X[i][1];
            X[i][7] = X[i][2] * X[i][2];
            y[i][0] = 0;
        } else {
            float t = (float) i / ((float) n);
            X[i][0] = 1;
            X[i][1] = (float) 0.5 - (a + b * t) * (float) cos((double) t * 10);
            X[i][2] = (float) 0.5 - (a + b * t) * (float) sin((double) t * 10);
            X[i][3] = (float) sin((double) (X[i][1] * 10));
            X[i][4] = (float) sin((double) (X[i][2] * 10));
            X[i][5] = X[i][2] * X[i][1];
            X[i][6] = X[i][1] * X[i][1];
            X[i][7] = X[i][2] * X[i][2];
            y[i][0] = 1;
        }
    }
}

void create_chesstable(float **X, float **y, int n, float dist) {
    int ok = 0;
    int class = rand() % 2;

    while (ok < n) {
        float a = rand_float();
        float b = rand_float();

        if (a > 0.5 + dist && b > 0.5 + dist) {
            X[ok][0] = 1;
            X[ok][1] = a;
            X[ok][2] = b;
            y[ok][0] = class == 0;
            ++ok;
        } else if (a > 0.5 + dist && b < 0.5 - dist) {
            X[ok][0] = 1;
            X[ok][1] = a;
            X[ok][2] = b;
            y[ok][0] = class == 1;
            ++ok;
        }  else if (a < 0.5 - dist && b > 0.5 + dist) {
            X[ok][0] = 1;
            X[ok][1] = a;
            X[ok][2] = b;
            y[ok][0] = class == 1;
            ++ok;
        } else if (a < 0.5 - dist && b < 0.5 - dist) {
            X[ok][0] = 1;
            X[ok][1] = a;
            X[ok][2] = b;
            y[ok][0] = class == 0;
            ++ok;
        }
    }
}

/* Splits the dataset into testing and training samples */
void split_train_test(float **X, float **y, float **X_train, float **X_test, float **y_train,
                        float **y_test, Dim dim, float ratio) {

    int split_size = dim.h * ratio;
    for (int i  = 0; i < dim.h; ++i) {
        if (i < split_size) {
            for (int j = 0; j < dim.w; ++j)
                X_train[i][j] = X[i][j];
            y_train[i] = y[i];
        } else {
            for (int j = 0; j < dim.w; ++j)
                X_test[i - split_size][j] = X[i][j];
            y_test[i - split_size] = y[i];
        }
    }
}
