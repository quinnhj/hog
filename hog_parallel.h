#include "readjpeg.h"
#include "clhelp.h"

void image_to_hist_2(cl_mem &image, cl_mem &hist, int width, int height,
            int cx, int cy, int n_cellsx, int n_cellsy, int num_orientations,
            cl_kernel &kernel, cl_command_queue &queue, cl_context &context);

/*
void hist_to_blocks_2(float *hist, float *normalised_blocks, int by, int bx,
            int n_blocksx, int n_blocksy, int num_orientations, int n_cellsx,
            int n_cellsy);
*/

