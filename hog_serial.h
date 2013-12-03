#include "readjpeg.h"

typedef struct
{
    float r;
    float g;
    float b;
} pixel_t;

double timestamp();
float rgb_to_grayscale(pixel_t input);
void convert_to_pixel(pixel_t *out, frame_ptr in);
void convert_to_frame(frame_ptr out, pixel_t *in);
void image_to_hist_serial(float *image, float *his, int width, int height,
            int cx, int cy, int n_cellsx, int n_cellsy, int num_orientations);
void image_to_gray_serial(pixel_t *inPix, float *pixels, int width, int height);
void hist_to_blocks_serial(float *hist, float *normalised_blocks, int by, int bx,
            int n_blocksx, int n_blocksy, int num_orientations, int n_cellsx,
            int n_cellsy);






