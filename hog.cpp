#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#include "readjpeg.h"

typedef struct
{
    float r;
    float g;
    float b;
} pixel_t;

double timestamp()
{
    struct timeval tv;
    gettimeofday (&tv, 0);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}

float rgb_to_lin(float x) {
    if (x < 0.04045) return x/12.92;
    return powf(((x+0.055)/1.055) , 2.4);
}

float lin_to_rgb(float y) {
    if (y <= 0.0031308) return 12.92 * y;
    return 1.055 * powf(y, 1/2.4) - 0.055;
}

float rgb_to_grayscale(pixel_t input){
    
    float r_lin = rgb_to_lin(input.r/255.0);
    float g_lin = rgb_to_lin(input.g/255.0);
    float b_lin = rgb_to_lin(input.b/255.0);
    float gray_lin = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin;

    return roundf(lin_to_rgb(gray_lin) * 255);
}


void convert_to_pixel(pixel_t *out, frame_ptr in)
{
    for(int y = 0; y < in->image_height; y++)
    {
        for(int x = 0; x < in->image_width; x++)
        {
            int r = (int)in->row_pointers[y][in->num_components*x + 0 ];
            int g = (int)in->row_pointers[y][in->num_components*x + 1 ];
            int b = (int)in->row_pointers[y][in->num_components*x + 2 ];
            out[y*in->image_width+x].r = (float)r;
            out[y*in->image_width+x].g = (float)g;
            out[y*in->image_width+x].b = (float)b;
     
        }
    }
}

void convert_to_frame(frame_ptr out, pixel_t *in)
{
    for(int y = 0; y < out->image_height; y++)
    {
        for(int x = 0; x < out->image_width; x++)
        {
            int r = (int)in[y*out->image_width + x].r;
            int g = (int)in[y*out->image_width + x].g;
            int b = (int)in[y*out->image_width + x].b;
            out->row_pointers[y][out->num_components*x + 0 ] = r;
            out->row_pointers[y][out->num_components*x + 1 ] = g;
            out->row_pointers[y][out->num_components*x + 2 ] = b;
        }
    }
}



int main(int argc, char *argv[])
{
    int c;
    char *inName = NULL;
    char *outName = NULL;
    int n=1;
    int width=-1,height=-1;
    frame_ptr frame;

    pixel_t *inPix=NULL;
    pixel_t *outPix=NULL;
    float *pixels=NULL;
    int *blur_radii = NULL;
    int nthreads = 1;

    srand(5);

    while((c = getopt(argc, argv, "i:n:o:t:"))!=-1)
    {
        switch(c)
        {
            case 'i':
                    inName = optarg;
                    break;
            case 'o':
                    outName = optarg;
                    break;
            case 'n':
                    n = atoi(optarg);
                    break;
            case 't':
                    nthreads = atoi(optarg);
                    break;
        }
    }

    
    
    if (inName == 0 || outName == 0)
    {
        printf("need input filename and output filename\n");
        return -1;
    }

    frame = read_JPEG_file(inName);
    if(!frame)
    {
        printf("unable to read %s\n", inName);
        exit(-1);
    }

    width = frame->image_width;
    height = frame->image_height;
 
    inPix = new pixel_t[width*height];
    outPix = new pixel_t[width*height];
    blur_radii = new int[width*height];
    pixels = new float[width*height];
 
    convert_to_pixel(inPix, frame);

    // Initialize to zeros gx and gy
    float *gx = NULL;
    float *gy = NULL;
    gx = new float[width*height];
    gy = new float[width*height];
    bzero(gx, sizeof(float)*width*height);
    bzero(gy, sizeof(float)*width*height);
    
    //Calculate Magnitude and Orientation
    float *magnitude = NULL;
    float *orientation = NULL;
    magnitude = new float[width*height];
    orientation = new float[width*height];
    bzero(magnitude, sizeof(float)*width*height);
    bzero(orientation, sizeof(float)*width*height);

    // Calculating sizes
    int sx, sy, cx, cy, bx, by;
    int pixels_per_cell = 8;
    int cells_per_block = 3;
    int num_orientations = 9;

    sx = width;
    sy = height;
    cx = cy = pixels_per_cell;
    bx = by = cells_per_block;
   
    // This may be off. Maybe it should be ceiling. 
    int n_cellsx = (int)floorf((sx / cx));
    int n_cellsy = (int)floorf((sy / cy));
   
    // Create histogram
    float *hist = new float[n_cellsx*n_cellsy*num_orientations];
    bzero(hist, sizeof(float)*n_cellsx*n_cellsy*num_orientations);

    // Compute rest of convolution
    float *temparr = new float[width*height];
 
 
    // Convert to grayscale and normalize

    /*
     * Fix loop ordering.
     * Merge gx and gy calculations into this.
     * Potentially merge orientation and magnitude calculations as well.
     * 
     * Parallelization: Probably done using tasks in openmp. OpenCl....nop.
     * Might need to precalculate some ghost data?
     *
    */

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            pixels[j*width + i] = sqrtf(rgb_to_grayscale(inPix[j*width + i]));
        }
    }

    printf("%f", pixels[rand() % 10]);
    
    // fill out gx and gy
    // gx
    // Merge up
    for (int i = 1; i < width; i++) {
        for (int j = 0; j < height; j++) {
            gx[j*width + i] = pixels[j*width + i] - pixels[j*width + i - 1];
        }
    }

    // gy
    // Merge up
    for (int i = 0; i < width; i++) {
        for (int j = 1; j < height; j++) {
            gy[j*width + i] = pixels[j*width + i] - pixels[(j-1)*width + i];
        }
    }
    printf("%f", gx[rand() % 10]);
    printf("%f", gy[rand() % 10]);

    // Merge up
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            magnitude[j*width+i] = sqrtf(powf(gx[j*width+i], 2) + powf(gy[j*width+i], 2));
            orientation[j*width+i] = atan2f(gy[j*width+i], gx[j*width+i] + 0.0000000001)
                    * (180 / 3.14159265) + 90;
        }
    }
    printf("%f", magnitude[rand() % 10]);
    printf("%f", orientation[rand() % 10]);

           
    // Move K into loop -- only do one nested for loop.
    // Needs to have block normalization inside, write to output array.
    for (int k = 0; k < num_orientations; k++) {
        bzero(temparr, sizeof(float)*width*height);

        for (int i = 0; i < width; i++) {
            for (int j = 0; i < height; j++) {
                bool in_bin = orientation[j*width + i] < ((180 / num_orientations)
                        * (k + 1));
                bool in_bin = in_bin && orientation[j*width + i] >=
                        ((180 / num_orientations) * k);

                if (in_bin) {
                    temparr[j*width + i] = magnitude[j*width + i];
                }
            }
        }
        
        uniform_filter(&temparr, cx, cy);

        for (int i = 0; i < n_cellsx; i++) {
            for (int j = 0; i < n_cellsy; j++) {
                int xval = cx/2 + cx*i;
                int yval = cy/2 + cy*j;
                hist[k*n_cellsx*n_cellsy + i*n_cellsx + j] = temparr[yval*width + xval];
            }
        }
    }

    double t0 = timestamp();
    omp_set_num_threads(nthreads);
    int flops = 0;

    t0 = timestamp() - t0;
    printf("%g sec\n", t0);
    printf("%d flops\n", flops);
    convert_to_frame(frame, outPix);

    write_JPEG_file(outName,frame,75);
    destroy_frame(frame);

    delete [] blur_radii;
    delete [] inPix; 
    delete [] outPix;
    return 0;
}



