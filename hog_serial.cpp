#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <iostream>
#include <fstream>

#include "hog_serial.h"
#include "readjpeg.h"
using namespace std;

void image_to_gray_serial(pixel_t *inPix, float *pixels, int width, int height) {
    #pragma omp parallel for
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            pixels[j*width + i] = sqrtf(rgb_to_grayscale(inPix[j*width + i]));
        }
    }
}


void image_to_hist_serial (float *image, float *hist, int width, int height,
                    int cx, int cy, int n_cellsx, int n_cellsy, int num_orientations) {

    /*
     * Logical steps:
     * 
     * For each pixel in image:
     *
     * 1) Calculate gx and gy based on neighbors. gx is the one to the right of it
     * minus itself. gy is the one below minus itself. If there is
     * no one to the right/below, set to zero. 
     *
     * 2) Calculate magnitude and orientation. Magnitude is defined as:
     *      sqrtf(powf(gx, 2) + powf(gy, 2));
     * Orientation is defined as:
     *      fmod(atan2f(gy, gx + 0.000000000001) * (180 / 3.14159265), 180);
     *
     * 3) Using orientation, calculate which "bin" it goes into. 
     *
     * 4) Figure out which cell it belongs in. Increment that by a
     * "spread out" (magnitude divided by filter size) value.
     */
    
    float gx;
    float gy;
    float orientation;
    float magnitude;
    int bin;

    float num_div_180 = (float)num_orientations / 180.0f;

    #pragma omp parallel for
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            
            // Step 1, calculating gx and gy
            if (i != width - 1) {
                gx = image[j*width + i + 1] - image[j*width + i];
            } else {
                gx = 0.0;
            }

            if (j != height - 1) {
                gy = image[(j+1)*width + i] - image[j * width + i];
            } else {
                gy = 0.0;
            }

            // Step 2, calculating mag and orientation
            magnitude = sqrtf(powf(gx, 2) + powf(gy, 2));
            orientation= fmod(atan2f(gy, gx + 0.00000000000001)
                    * (180 / 3.14159265), 180);
            if (orientation < 0) {
                orientation += 180;
            }

            // Step 3, calculating bin.
            bin = (int)floor(orientation * num_div_180);

            // Step 4, calculating which cell it belongs to
            
            int cellx = i / cx;
            int celly = j / cy;
            
            hist[celly*n_cellsx*num_orientations + cellx*num_orientations + bin] += 
                        magnitude / (cx * cy);

        }
    }
}

void hist_to_blocks_serial(float *hist, float *normalised_blocks, int by, int bx,
            int n_blocksx, int n_blocksy, int num_orientations, int n_cellsx,
            int n_cellsy) {

    //Normalizing into flat block array
    float eps = 1e-5;
    float arr_sum = 0; 
    int block_size = by * bx * num_orientations;
    #pragma omp parallel for
    for (int j = 0; j < n_blocksy; j++) {
        for (int i = 0; i < n_blocksx; i++) {
            arr_sum = 0;

            for (int ay = 0; ay < by; ay++) {
                for (int ax = 0; ax < bx; ax++) {
                    for (int k = 0; k < num_orientations; k++) {
                        arr_sum += hist[(j+ay)*n_cellsx*num_orientations
                                    + (i+ax)*num_orientations + k];
                    }
                }
            }

            for (int ay = 0; ay < by; ay++) {
                for (int ax = 0; ax < bx; ax++) {
                    for (int k = 0; k < num_orientations; k++) {
                        normalised_blocks[j*n_blocksx*block_size +
                                    i*block_size + ay*bx*num_orientations
                                    + ax*num_orientations + k] =

                        hist[(j+ay)*n_cellsx*num_orientations
                                + (i+ax)*num_orientations + k] / sqrtf(powf(arr_sum, 2) + eps);
                    }
                }
            }
        }
    }

}


