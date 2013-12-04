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
#include <string>
#include <cmath>
#include <cassert>

#include "readjpeg.h"
#include "clhelp.h"
#include "hog_serial.h"
#include "hog_parallel.h"
using namespace std;

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


int main(int argc, char *argv[])
{    
    int c;
    char *inName = NULL;
    int width=-1,height=-1;
    frame_ptr frame;
    int version = 1;

    pixel_t *inPix=NULL;
    float *pixels=NULL;
    int nthreads = 1;
   
    while((c = getopt(argc, argv, "i:t:v:"))!=-1)
    {
        switch(c)
        {
            case 'i':
                    inName = optarg;
                    break;
            case 't':
                    nthreads = atoi(optarg);
                    break;
            case 'v':
                    version = atoi(optarg);
                    break;
        }
    }

    if (inName == 0)
    {
        printf("need input filename\n");
        return -1;
    }
 
    int num_timestamps = 5;
    double timestamps[num_timestamps];
    timestamps[0] = timestamp();
    omp_set_num_threads(nthreads);

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
    
    int n_blocksx = (n_cellsx - bx) + 1;
    int n_blocksy = (n_cellsy - by) + 1;
    int block_arr_size = n_blocksx*n_blocksy*by*bx*num_orientations;
    float *normalised_blocks = new float[block_arr_size];
    bzero(normalised_blocks, sizeof(float)*block_arr_size);

    /*
     * Initializing opencl
     *
     */
    std::string kernel_source_str;
    std::string arraycompact_kernel_file =
            std::string("hog_parallel.cl");

    std::list<std::string> kernel_names;
    std::string hist2_name_str = std::string("image_to_hist_2");
    std::string block2_name_str = std::string("hist_to_blocks_2");
    std::string hist3_name_str = std::string("image_to_hist_3");
    std::string block3_name_str = std::string("hist_to_blocks_3");

    kernel_names.push_back(hist2_name_str);
    kernel_names.push_back(block2_name_str);
    kernel_names.push_back(hist3_name_str);
    kernel_names.push_back(block3_name_str);

    cl_vars_t cv;
    std::map<std::string, cl_kernel> kernel_map;

    readFile(arraycompact_kernel_file, kernel_source_str);
    initialize_ocl(cv);

    compile_ocl_program(kernel_map, cv,
            kernel_source_str.c_str(),
            kernel_names);

    cl_mem g_pixels, g_hist, g_normalised_blocks;

    cl_int err = CL_SUCCESS;
    g_pixels = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
            sizeof(float)*width*height, NULL, &err);
    CHK_ERR(err);
    g_hist = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
            sizeof(float)*n_cellsx*n_cellsy*num_orientations, NULL, &err);
    CHK_ERR(err);
    g_normalised_blocks = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
            sizeof(float)*block_arr_size, NULL, &err);
    CHK_ERR(err);

    size_t global_work_size[2] = {n_cellsy, n_cellsx};
    size_t local_work_size[2] = {cy, cx};

    //Timestamp for end of setup. 
    timestamps[1] = timestamp();

    /* Versions:
     * 
     * 1) Serial
     *
     * 2) First pass Parallel
     *
     * 3) Parallel Kernel with Approximations
     *
     *
     *
     *
     *
     *
     *
     *
     *
     *
     *
     *
     *
     *
     *
     */

    switch (version) {
        case 1:
            image_to_gray_serial(inPix, pixels, width, height);
            break;
    
        default:
            image_to_gray_serial(inPix, pixels, width, height);
            break;
    }

    err = clEnqueueWriteBuffer(cv.commands, g_pixels, true, 0,
            sizeof(float)*width*height, pixels, 0, NULL, NULL);
    CHK_ERR(err);
    
    //TS for converting to grayscale
    timestamps[2] = timestamp();
    double time_reading = 0.0;

    //Calculate histogram
    switch (version) {
        case 1:
            image_to_hist_serial(pixels, hist, width, height,
                    cx, cy, n_cellsx, n_cellsy, num_orientations);
            break;
   
        case 2:
            image_to_hist_parallel(g_pixels, g_hist, width, height, cx, cy,
                    n_cellsx, n_cellsy, num_orientations, kernel_map[hist2_name_str],
                    cv.commands, cv.context);
       
            err = clFlush(cv.commands);
            CHK_ERR(err);
             
            
            time_reading = timestamp();
            err = clEnqueueReadBuffer(cv.commands, g_hist, true, 0,
                    sizeof(float) * n_cellsx * n_cellsy * num_orientations,
                    hist, 0, NULL, NULL);
            CHK_ERR(err);
            time_reading = time_reading - timestamp();
            break; 

        case 3:
            image_to_hist_parallel(g_pixels, g_hist, width, height, cx, cy,
                    n_cellsx, n_cellsy, num_orientations, kernel_map[hist3_name_str],
                    cv.commands, cv.context);
       
            err = clFlush(cv.commands);
            CHK_ERR(err);
             
            
            time_reading = timestamp();
            err = clEnqueueReadBuffer(cv.commands, g_hist, true, 0,
                    sizeof(float) * n_cellsx * n_cellsy * num_orientations,
                    hist, 0, NULL, NULL);
            CHK_ERR(err);
            time_reading = time_reading - timestamp();
            break; 

        default:
            image_to_hist_serial(pixels, hist, width, height,
                    cx, cy, n_cellsx, n_cellsy, num_orientations);
            break;
    }

    // TS for finished histogram
    timestamps[3] = timestamp();
    double temp = 0.0;
    
    //Calculate normalised blocks
    switch (version) {
        case 1:
            hist_to_blocks_serial(hist, normalised_blocks, by, bx,
                    n_blocksx, n_blocksy, num_orientations,n_cellsx,
                    n_cellsy);
            break;

        case 2:
            hist_to_blocks_parallel(g_hist, g_normalised_blocks, by, bx, n_blocksx, n_blocksy,
                    num_orientations, n_cellsx, n_cellsy, kernel_map[block2_name_str],
                    cv.commands, cv.context);
       
            err = clFlush(cv.commands);
            CHK_ERR(err);
            
            temp = timestamp(); 
            err = clEnqueueReadBuffer(cv.commands, g_normalised_blocks, true, 0,
                    sizeof(float) * n_blocksx * n_blocksy * bx * by *num_orientations,
                    normalised_blocks, 0, NULL, NULL);
            CHK_ERR(err);
            temp = temp - timestamp();
            time_reading += temp; 
            break;

        case 3:
            hist_to_blocks_parallel3(g_hist, g_normalised_blocks, by, bx, n_blocksx, n_blocksy,
                    num_orientations, n_cellsx, n_cellsy, kernel_map[block3_name_str],
                    cv.commands, cv.context);
       
            err = clFlush(cv.commands);
            CHK_ERR(err);
            
            temp = timestamp(); 
            err = clEnqueueReadBuffer(cv.commands, g_normalised_blocks, true, 0,
                    sizeof(float) * n_blocksx * n_blocksy * bx * by *num_orientations,
                    normalised_blocks, 0, NULL, NULL);
            CHK_ERR(err);
            temp = temp - timestamp();
            time_reading += temp; 
            break;

        default:
            hist_to_blocks_serial(hist, normalised_blocks, by, bx,
                    n_blocksx, n_blocksy, num_orientations,n_cellsx,
                    n_cellsy);
            break;
    }

    // TS to flatten
    timestamps[4] = timestamp(); 

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
    //printf("Timestamps:\n\n");
    double total_time = timestamps[num_timestamps-1] - timestamps[2] - time_reading;
    printf("%f,", total_time);
    for(int i = 1; i < num_timestamps; i++) {
        printf("%f,%f,", timestamps[i] - timestamps[i-1], ((timestamps[i] - timestamps[i-1]) / total_time) * 100);
    }

    //Free up stuff
    clReleaseMemObject(g_hist);
    clReleaseMemObject(g_pixels);
    clReleaseMemObject(g_normalised_blocks);

    destroy_frame(frame);
    delete [] inPix; 
    delete [] hist;
    delete [] normalised_blocks;
    delete [] pixels;
    return 0;
}



