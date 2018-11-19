/*
 * This file contains the definitions of the very basic functions
 * that is used to create datasets, and implement ANNs in C
 *
 * Made by Tamás Imets
 * Date: 18th of November, 2018
 * Version: 0.1
 *
 */

#include "perceptron.h"


/* Terminates program */
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

void create_checktable(float **X, float **y, int n, float dist) {
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




