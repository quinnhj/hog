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

#include "readjpeg.h"
using namespace std;

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

float rgb_to_grayscale(pixel_t input){
    return 0.299*input.r + 0.587*input.g + 0.114*input.b;
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

void image_to_hist (float *image, float *hist, int width, int height,
                    int cx, int cy, int n_cellsx, int n_cellsy, int num_orientations) {

    /*
     * Logical steps:
     * 
     * For each pixel in image:
     *
     * 1) Calculate gx and gy based on neighbors. gx is itself minus the
     * one to the left of it. gy is itself minus the one above it. If there is
     * no one to the left/above, set to zero. 
     *
     * 2) Calculate magnitude and orientation. Magnitude is defined as:
     *      sqrtf(powf(gx, 2) + powf(gy, 2));
     * Orientation is defined as:
     *      fmod(atan2f(gy, gx + 0.000000000001) * (180 / 3.14159265), 180);
     *
     * 3) Using orientation, calculate which "bin" it goes into. 
     *
     * 4) Figure out which cells it belongs in. Increment those by a
     * "spread out" (magnitude divided by filter size) value.
     *
     * To figure out which cell it belongs to, know that uniform filters
     * distribute to anything withing a square of "radius" cx and cy around
     * it. So if we're smart, we can do mod math on the coordinates to figure
     * them out.
     *
     *
     *
     *
     *
     */
    
    float gx;
    float gy;
    float orientation;
    float magnitude;
    int bin;

    float num_div_180 = (float)num_orientations / 180.0f;

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


    /*
     * Declaration of timestamp constant / array
     */
      
    int num_timestamps = 5;
    double timestamps[num_timestamps];
    timestamps[0] = timestamp();

    /*
     * Reading in the JPEG image
     * Does validation and convert into a pixel form
     */

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

    /*
     * Here we declare and create all data structures
     * This is also where constants are set, such as 
     * pixels per cell and whatnot. Those should probably
     * be done via command line input later.
     */

    int sx, sy, cx, cy, bx, by;
    int pixels_per_cell = 8;
    int cells_per_block = 3;
    int num_orientations = 9;

    sx = width;
    sy = height;
    cx = cy = pixels_per_cell;
    bx = by = cells_per_block;
   
    int n_cellsx = (int)floorf((sx / cx));
    int n_cellsy = (int)floorf((sy / cy));
   
    float *hist = new float[n_cellsx*n_cellsy*num_orientations];
    bzero(hist, sizeof(float)*n_cellsx*n_cellsy*num_orientations);

    //Timestamp for end of setup. 
    timestamps[1] = timestamp();
   
    // Convert to grayscale and normalize
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            pixels[j*width + i] = sqrtf(rgb_to_grayscale(inPix[j*width + i]));
        }
    }

    //Finished converting to grayscale
    timestamps[2] = timestamp();
    
    image_to_hist(pixels, hist, width, height,
                    cx, cy, n_cellsx, n_cellsy, num_orientations);

    // TS for finished histogram
    timestamps[3] = timestamp();

    int n_blocksx = (n_cellsx - bx) + 1;
    int n_blocksy = (n_cellsy - by) + 1;
    int block_arr_size = n_blocksx*n_blocksy*by*bx*num_orientations;
    float *normalised_blocks = new float[block_arr_size];
    bzero(normalised_blocks, sizeof(float)*block_arr_size);

    //Normalizing into flat block array
    float eps = 1e-5;
    float arr_sum = 0;
   
    int block_size = by * bx * num_orientations;
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

    // Time to flatten
    timestamps[4] = timestamp();

    omp_set_num_threads(nthreads);
    int flops = 0;
    convert_to_frame(frame, outPix);

    write_JPEG_file(outName,frame,75);
    destroy_frame(frame);


    //Saving to text file:
    ofstream file;
    file.open("output/cpp_out.txt");
    for (int i = 0; i < block_arr_size; i++) {
        file << normalised_blocks[i];
        file << "\n";
    }
    file.close();

    //Saving to text file:
    file.open("output/cpp_hist.txt");
    for (int i = 0; i < n_cellsx*n_cellsy*num_orientations; i++) {
        file << hist[i];
        file << "\n";
    }
    file.close();

    //Print timestamping data:
    printf("Timestamps:\n\n");
    double total_time = timestamps[num_timestamps-1] - timestamps[0];
    printf("Total time: %f seconds\n", total_time);
    for (int i = 1; i < num_timestamps; i++) {
        printf("%d: %f\n", i, timestamps[i]);
        printf("\t%f percent\n", ((timestamps[i] - timestamps[i-1]) / total_time) * 100);
    }

    delete [] blur_radii;
    delete [] inPix; 
    delete [] outPix;
    delete [] hist;
    delete [] normalised_blocks;
    return 0;
}



