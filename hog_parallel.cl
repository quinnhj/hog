
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
    //size_t idx = get_global_id(0);
    //size_t tid = get_local_id(0);
    //size_t dim = get_local_size(0);

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
        //hist[celly*n_cellsx*num_orientations + cellx*num_orientations + id] = (float) 111;
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

        //val = hist[group_offset + id];
        val = hist[group_offset + id_by*n_cellsx*num_orientations
                    + id_bx * num_orientations + id_num];

        buf[id] = val;
    } else {
        buf[id] = 0.0;
    }
   
    barrier(CLK_LOCAL_MEM_FENCE);

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







