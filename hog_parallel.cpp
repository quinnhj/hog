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
#include <cassert>
#include <cmath>
#include <string>

#include "clhelp.h"
#include "hog_serial.h"
#include "hog_parallel.h"
#include "readjpeg.h"
using namespace std;



void image_to_hist_2 (cl_mem &image, cl_mem &hist, int width, int height,
                    int cx, int cy, int n_cellsx, int n_cellsy, int num_orientations,
                    cl_kernel &kernel, cl_command_queue &queue, cl_context &context) {

    size_t global_work_size[2] = {n_cellsy*cy, n_cellsx*cx};
    size_t local_work_size[2] = {cy, cx};
    
    cl_int err = CL_SUCCESS;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &hist);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 2, num_orientations*sizeof(float), NULL);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 3, sizeof(int), &width);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 4, sizeof(int), &height);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 5, sizeof(int), &cx);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 6, sizeof(int), &cy);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 7, sizeof(int), &n_cellsx);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 8, sizeof(int), &n_cellsy);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 9, sizeof(int), &num_orientations);
    CHK_ERR(err);

    err = clEnqueueNDRangeKernel(queue,
                kernel,
                2,
                NULL,
                global_work_size,
                local_work_size,
                0,
                NULL,
                NULL
                );
    CHK_ERR(err);

    err = clFinish(queue);
    CHK_ERR(err);

}


void hist_to_blocks_2(cl_mem &hist, cl_mem &normalised_blocks, int by, int bx,
            int n_blocksx, int n_blocksy, int num_orientations, int n_cellsx,
            int n_cellsy, cl_kernel &kernel, cl_command_queue &queue, 
            cl_context &context) {
    
    unsigned int size = by*bx*num_orientations;

    //Shift up! Yay! To nearest power of 2.
    size--;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    size++;
    printf("Size: %d\n", size);

    size_t global_work_size[1] = {size*n_blocksx*n_blocksy};
    size_t local_work_size[1] = {size};
    
    cl_int err = CL_SUCCESS;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &hist);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &normalised_blocks);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 2, size*sizeof(float), NULL);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 3, sizeof(int), &by);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 4, sizeof(int), &bx);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 5, sizeof(int), &n_blocksx);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 6, sizeof(int), &n_blocksy);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 7, sizeof(int), &num_orientations);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 8, sizeof(int), &n_cellsx);
    CHK_ERR(err);

    err = clSetKernelArg(kernel, 9, sizeof(int), &n_cellsy);
    CHK_ERR(err);

    err = clEnqueueNDRangeKernel(queue,
                kernel,
                1,
                NULL,
                global_work_size,
                local_work_size,
                0,
                NULL,
                NULL
                );
    CHK_ERR(err);

    err = clFinish(queue);
    CHK_ERR(err);


}


