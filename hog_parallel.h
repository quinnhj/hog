#include "readjpeg.h"
#include "clhelp.h"

void image_to_gray_parallel(pixel_t *inPix, float *pixels, int width, int height,
            int cx, int cy, int n_cellsx, int n_cellsy);

void image_to_hist_parallel(cl_mem &image, cl_mem &hist, int width, int height,
            int cx, int cy, int n_cellsx, int n_cellsy, int num_orientations,
            cl_kernel &kernel, cl_command_queue &queue, cl_context &context);

void image_to_hist_parallel5(cl_mem &image, cl_mem &hist, int width, int height,
            int cx, int cy, int n_cellsx, int n_cellsy, int num_orientations, int by,
            cl_kernel &kernel, cl_command_queue &queue, cl_context &context);

void hist_to_blocks_parallel(cl_mem &hist, cl_mem &normalised_blocks, int by, int bx,
            int n_blocksx, int n_blocksy, int num_orientations, int n_cellsx,
            int n_cellsy, cl_kernel &kernel, cl_command_queue &queue, 
            cl_context &context);

void hist_to_blocks_parallel3(cl_mem &hist, cl_mem &normalised_blocks, int by, int bx,
            int n_blocksx, int n_blocksy, int num_orientations, int n_cellsx,
            int n_cellsy, cl_kernel &kernel, cl_command_queue &queue, 
            cl_context &context);

void hist_to_blocks_parallel5(cl_mem &hist, cl_mem &normalised_blocks, int by, int bx,
            int n_blocksx, int n_blocksy, int num_orientations, int n_cellsx,
            int n_cellsy, cl_kernel &kernel, cl_command_queue &queue, 
            cl_context &context);

