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




