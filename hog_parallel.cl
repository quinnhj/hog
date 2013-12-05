#define HALF_PI 1.5707963267948966f
#define ONEEIGHTY_PI 57.29577951308232f

inline void AtomicAdd(volatile __local float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __local unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline float fast_arctan_degree (float y, float x) {
    
    float angle;
    float z = y/x;
    float tmp = z < 0.0f ? -z : z;
    if (tmp < 1.0f ) {
        angle = (z/(1.0f + 0.28f*z*z)) * ONEEIGHTY_PI;
    } else {
        angle = (HALF_PI - z/(z*z + 0.28f)) * ONEEIGHTY_PI;
    }
    return angle + 360;
}

/* Not used
inline float fast_fmod(float x, float y) {
    float a = x/y;
    return (a-(int)a) * y;   
} 
*/   

/*****************************************************************************
 * Version 2
 ****************************************************************************/

__kernel void image_to_hist_2(
        __global float *image,
        __global float *hist,
        __local float *buf,
        int width,
        int height,
        int cx, 
        int cy,
        int n_cellsx,
        int n_cellsy,
        int num_orientations)

{
    int cellx = get_group_id(1);
    int celly = get_group_id(0);
    size_t i = get_global_id(1);
    size_t j = get_global_id(0);

    float gx;
    float gy;
    float orientation;
    float magnitude;
    int bin;
    
    // Step 0, initialize buf
    int id = get_local_id(0) * get_local_size(1) + get_local_id(1);
    if (id < num_orientations) {
        buf[id] = 0;
    }

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
    magnitude = sqrt(pow(gx, 2) + pow(gy, 2));
    orientation= fmod(atan2(gy, gx + 0.00000000000001)
            * (180 / 3.14159265), 180);
    if (orientation < 0) {
        orientation += 180;
    }

    // Step 3, calculating bin.
    bin = (int)floor(orientation * ((float)num_orientations / 180.0f));

    // Step 4, atomic add to buffer...sketch.
    AtomicAdd(&buf[bin], magnitude / (cx * cy));

    // Barrier to wait for all threads to finish calculating the bins.
    barrier(CLK_LOCAL_MEM_FENCE);

    if (id < num_orientations) {
        hist[celly*n_cellsx*num_orientations + cellx*num_orientations + id] = 
                    buf[id];
    }
}


__kernel void hist_to_blocks_2(
        __global float *hist,
        __global float *normalised_blocks,
        __local float *buf,
        int by,
        int bx,
        int n_blocksx, 
        int n_blocksy,
        int num_orientations,
        int n_cellsx,
        int n_cellsy)
{

    size_t dim = get_local_size(0);
    int i = get_group_id(0) % n_blocksx;
    int j = get_group_id(0) / n_blocksx;

    int group_offset = j*n_cellsx*num_orientations + i*num_orientations;
    int id = get_local_id(0);
    
    float val = 0.0;
    int block_size = by*bx*num_orientations;
    int id_num = id % num_orientations;
    int id_by = id / (num_orientations * bx);
    int id_bx = (id / num_orientations) % bx;

    if (id < block_size) {
        val = hist[group_offset + id_by*n_cellsx*num_orientations
                    + id_bx * num_orientations + id_num];
        buf[id] = val;
    } else {
        buf[id] = 0.0;
    }
   
    barrier(CLK_LOCAL_MEM_FENCE);

    //Reduce to buf[0]
    for (int s = dim/2; s > 0; s>>= 1) {
        if (id < s) {
            buf[id] += buf[id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float sum = buf[0];
    if (id < block_size) {
        normalised_blocks[j*n_blocksx*block_size +
                i*block_size + id] = val / sum;
    }    
}

/*****************************************************************************
 * Version 3
 ****************************************************************************/

__kernel void image_to_hist_3(
        __global float *image,
        __global float *hist,
        __local float *buf,
        int width,
        int height,
        int cx, 
        int cy,
        int n_cellsx,
        int n_cellsy,
        int num_orientations)

{
    int cellx = get_group_id(1);
    int celly = get_group_id(0);
    size_t i = get_global_id(1);
    size_t j = get_global_id(0);

    float gx = 0.0;
    float gy = 0.0;
    float orientation;
    float magnitude;
    int bin;
    
    // Step 0, initialize buf
    int id = get_local_id(0) * get_local_size(1) + get_local_id(1);
    if (id < num_orientations) {
        buf[id] = 0;
    }

    // Step 1, calculating gx and gy
    if (i != width - 1) {
        gx = image[j*width + i + 1] - image[j*width + i];
    }

    if (j != height - 1) {
        gy = image[(j+1)*width + i] - image[j * width + i];
    }

    
    // Step 2, calculating mag and orientation
    magnitude = sqrt(gx*gx + gy*gy);
    orientation = fast_arctan_degree(gy, gx + 0.00000000000001) / 180;
    orientation = (orientation - (int) orientation) * 180;

    // Step 3, calculating bin.
    bin = (int)floor(orientation * ((float)num_orientations / 180.0f));

    // Step 4, atomic add to buffer...sketch.
    AtomicAdd(&buf[bin], magnitude / (cx * cy));

    // Barrier to wait for all threads to finish calculating the bins.
    barrier(CLK_LOCAL_MEM_FENCE);

    if (id < num_orientations) {
        hist[celly*n_cellsx*num_orientations + cellx*num_orientations + id] = 
                    buf[id];
    }
}


__kernel void hist_to_blocks_3(
        __global float *hist,
        __global float *normalised_blocks,
        __local float *buf,
        int block_size,
        int bx,
        int n_blocksx, 
        int num_orientations,
        int n_cellsx_num)
{

    int i = get_group_id(0) % n_blocksx;
    int j = get_group_id(0) / n_blocksx;

    int group_offset = j*n_cellsx_num + i*num_orientations;
    int id = get_local_id(0);
    
    float val;
    buf[id] = 0.0;
    if (id < block_size) {
        val = hist[group_offset +  (id / (num_orientations * bx))*n_cellsx_num
                    + ((id / num_orientations) % bx) * num_orientations 
                    + (id % num_orientations)];
        buf[id] = val;
    }
   
    barrier(CLK_LOCAL_MEM_FENCE);

    //Reduce to buf[0]
    for (int s = get_local_size(0)/2; s > 0; s>>= 1) {
        if (id < s) {
            buf[id] += buf[id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float sum = buf[0];
    if (id < block_size) {
        normalised_blocks[ j*n_blocksx*block_size +
                i*block_size + id] = val / sum;
    }    
}


/*****************************************************************************
 * Version 4
 ****************************************************************************/

__kernel void image_to_hist_4(
        __global float *image,
        __global float *hist,
        __local float *buf,
        int width,
        int height,
        int cx, 
        int cy,
        int n_cellsx,
        int n_cellsy,
        int num_orientations)

{
    int cellx = get_group_id(1);
    int celly = get_group_id(0);
    size_t i = get_global_id(1);
    size_t j = get_global_id(0);

    float gx = 0.0;
    float gy = 0.0;
    float orientation;
    float magnitude;
    int bin;
    
    // Step 0, initialize buf
    int id = get_local_id(0) * get_local_size(1) + get_local_id(1);
    if (id < num_orientations) {
        buf[id] = 0;
    }

    int size = cx * cy;
    int gmin = ((j/cy)*n_cellsx + (i/cx)) * size +
                (j % cy)*cx + (i % cx);

    // Step 1, calculating gx and gy
    if (i != width - 1) {
        gx = image[((j/cy)*n_cellsx + ((i+1)/cx)) * size +
                (j % cy)*cx + ((i+1) % cx)] - image[gmin];
    }

    if (j != height - 1) {
        gy = image[(((j+1)/cy)*n_cellsx + (i/cx)) * size +
                ((j+1) % cy)*cx + (i % cx)] - image[gmin];
    }

    
    // Step 2, calculating mag and orientation
    magnitude = sqrt(gx*gx + gy*gy);
    orientation = fast_arctan_degree(gy, gx + 0.00000000000001) / 180;
    orientation = (orientation - (int) orientation) * 180;

    // Step 3, calculating bin.
    bin = (int)floor(orientation * ((float)num_orientations / 180.0f));

    // Step 4, atomic add to buffer...sketch.
    AtomicAdd(&buf[bin], magnitude / (cx * cy));

    // Barrier to wait for all threads to finish calculating the bins.
    barrier(CLK_LOCAL_MEM_FENCE);

    if (id < num_orientations) {
        hist[celly*n_cellsx*num_orientations + cellx*num_orientations + id] = 
                    buf[id];
    }
}


__kernel void hist_to_blocks_4(
        __global float *hist,
        __global float *normalised_blocks,
        __local float *buf,
        int block_size,
        int bx,
        int n_blocksx, 
        int num_orientations,
        int n_cellsx_num)
{

    int i = get_group_id(0) % n_blocksx;
    int j = get_group_id(0) / n_blocksx;

    int group_offset = j*n_cellsx_num + i*num_orientations;
    int id = get_local_id(0);
    
    float val;
    buf[id] = 0.0;
    if (id < block_size) {
        val = hist[group_offset +  (id / (num_orientations * bx))*n_cellsx_num
                    + ((id / num_orientations) % bx) * num_orientations 
                    + (id % num_orientations)];
        buf[id] = val;
    }
   
    barrier(CLK_LOCAL_MEM_FENCE);

    //Reduce to buf[0]
    for (int s = get_local_size(0)/2; s > 0; s>>= 1) {
        if (id < s) {
            buf[id] += buf[id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float sum = buf[0];
    if (id < block_size) {
        normalised_blocks[ j*n_blocksx*block_size +
                i*block_size + id] = val / sum;
    }    
}
















