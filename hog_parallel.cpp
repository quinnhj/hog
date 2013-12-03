#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include "clhelp.h"

// Just a header for a function that is defined below
void histogram_helper(cl_command_queue &queue,
		    cl_context &context,
		    cl_kernel &hist_kern,
		    cl_mem &in, 
		    cl_mem &out);

int main(int argc, char *argv[])
{
  std::string kernel_source_str;
 
  /* Provide names of the OpenCL kernels
   * and cl file that they're kept in */ 
  std::string arraycompact_kernel_file = 
    std::string("hog_parallel.cl");
  
  std::list<std::string> kernel_names;
  std::string histogram_name_str = std::string("compute_histogram");
  kernel_names.push_back(histogram_name_str);

  cl_vars_t cv; 
  
  std::map<std::string, cl_kernel> 
    kernel_map;

  int c;
  int n = (1<<20); // default size of the array
  int *in, *out;
  int *c_scan;
  int n_out=-1;
  bool silent = false;

  // Just processing command-line arguments (size and silent)
  while((c = getopt(argc, argv, "n:s:"))!=-1)
  {
    switch(c)
	  {
	    case 'n':
	      n = 1 << atoi(optarg);
	      break;
	    case 's':
	      silent = atoi(optarg) == 1;
	      break;
	  }
  }

  /* Allocate arrays on the host
   * and fill with data */
  in = new int[n];
  out = new int[n];
  c_scan = new int[n];
 
  bzero(out, sizeof(int)*n);
  bzero(c_scan, sizeof(int)*n);

  srand(5);
  for(int i = 0; i < n; i++)
  {
    in[i] = rand();
  }

  /* Read OpenCL file into STL string */
  readFile(arraycompact_kernel_file,
	   kernel_source_str);
  
  /* Initialize the OpenCL runtime 
     Source in clhelp.cpp */
  initialize_ocl(cv);

  /* Compile all OpenCL kernels */
  compile_ocl_program(kernel_map, cv, 
		      kernel_source_str.c_str(),
		      kernel_names);
  
  /* Arrays on the device (GPU) */
  cl_mem g_in, g_out;
  cl_mem g_temp;
  
  // Allocate Buffers on the GPU
  cl_int err = CL_SUCCESS;
  g_in = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
		       sizeof(int)*n,NULL,&err);
  CHK_ERR(err);  

  g_temp = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
			  sizeof(int)*n,NULL,&err);
  CHK_ERR(err);
  
  g_out = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
			 sizeof(int)*n,NULL,&err);
  CHK_ERR(err);
  
  //copy data from host CPU to GPU
  err = clEnqueueWriteBuffer(cv.commands, g_in, true, 0, sizeof(int)*n,
			     in, 0, NULL, NULL);
  CHK_ERR(err);

  err = clEnqueueWriteBuffer(cv.commands, g_out, true, 0, sizeof(int)*n,
			     c_scan, 0, NULL, NULL);
  CHK_ERR(err);
  
  // Set global and local work size respectively.
  size_t global_work_size[1] = {n};
  size_t local_work_size[1] = {128};

  // This function is provided to us to ensure that the global size
  // is divisible by the local size. 
  adjustWorkSize(global_work_size[0], local_work_size[0]);
  global_work_size[0] = std::max(local_work_size[0], global_work_size[0]);
  int left_over = 0;

  double t0 = timestamp();
  
  histogram_helper(cv.commands, cv.context,
    kernel_map[histogram_name_str],
    kernel_map[update_name_str],
    g_temp,
    g_out,
    1,
    i,
    n);

    // Set argument for reassemble kernel: g_temp is input
    err = clSetKernelArg(kernel_map[reassemble_name_str], 0, sizeof(cl_mem), &g_temp);
    CHK_ERR(err);
      

    // Set argument for reassemble kernel: scan kernel's output
    err = clSetKernelArg(kernel_map[reassemble_name_str], 2, sizeof(cl_mem), &g_out);
    CHK_ERR(err);

    // Set argument for reassemble kernel: local memory for the kernel to play with
    err = clSetKernelArg(kernel_map[reassemble_name_str], 3, 2*local_work_size[0]*sizeof(cl_int), NULL);
    CHK_ERR(err);

    // Set argument for reassemble kernel: what bit we are looking at ("k")
    err = clSetKernelArg(kernel_map[reassemble_name_str], 4, sizeof(int), &i);
    CHK_ERR(err);

    // Set argument for reassemble kernel: size of the input
    err = clSetKernelArg(kernel_map[reassemble_name_str], 5, sizeof(int), &n);
    CHK_ERR(err);

    // Execute reassemble kernel
    err = clEnqueueNDRangeKernel(cv.commands,
               kernel_map[reassemble_name_str],
               1,//work_dim,
               NULL, //global_work_offset
               global_work_size, //global_work_size
               local_work_size, //local_work_size
               0, //num_events_in_wait_list
               NULL, //event_wait_list
               NULL //
               );
    CHK_ERR(err);
  }

  // Shut down the OpenCL runtime
  clReleaseMemObject(g_in); 
  clReleaseMemObject(g_out);
  clReleaseMemObject(g_temp);
  
  uninitialize_ocl(cv);

  delete [] in;
  delete [] out;
  delete [] c_scan;
  return 0;
}

void histogram_helper(cl_command_queue &queue,
		cl_context &context,
		cl_kernel &scan_kern,
		cl_kernel &update_kern,
		cl_mem &in, 
		cl_mem &out, 
		int v,
		int k,
		int len)
{
  // Set global and local work size respectively.
  size_t global_work_size[1] = {len};
  size_t local_work_size[1] = {128};
  int left_over = 0;
  cl_int err;
  
  // This function is provided to us to ensure that the global size
  // is divisible by the local size.
  adjustWorkSize(global_work_size[0], local_work_size[0]);
  global_work_size[0] = std::max(local_work_size[0], global_work_size[0]);

  // This variable is crucial to figure out whether we need to perform
  // further recursive calls to recursive_scan or whether the array is
  // small enough that it can be processed by one scan kernel call (left_over = 1). 
  left_over = global_work_size[0] / local_work_size[0];
  
  // Allocates a buffer on the GPU
  cl_mem g_bscan = clCreateBuffer(context,CL_MEM_READ_WRITE, 
				  sizeof(int)*left_over,NULL,&err);
  CHK_ERR(err);

  // This basically just sets all the kernel arguments.
  err = clSetKernelArg(scan_kern, 0, sizeof(cl_mem), &in);
  CHK_ERR(err);

  err = clSetKernelArg(scan_kern, 1, sizeof(cl_mem), &out);
  CHK_ERR(err);

  /* CS194: Per work-group partial scan output */
  err = clSetKernelArg(scan_kern, 2, sizeof(cl_mem), &g_bscan);
  CHK_ERR(err);

  /* CS194: number of bytes for dynamically 
   * sized local (private memory) "buf"*/
  err = clSetKernelArg(scan_kern, 3, 2*local_work_size[0]*sizeof(cl_int), NULL);
  CHK_ERR(err);

  /* CS194: v will be either 0 or 1 in order to perform
   * a scan of bits set (or unset) */
  err = clSetKernelArg(scan_kern, 4, sizeof(int), &v);
  CHK_ERR(err);

  /* CS194: the current bit position (0 to 31) that
   * we want to operate on */
  err = clSetKernelArg(scan_kern, 5, sizeof(int), &k);
  CHK_ERR(err);

  // size of input array being processed
  err = clSetKernelArg(scan_kern, 6, sizeof(int), &len);
  CHK_ERR(err);

  // Launch scan kernel
  err = clEnqueueNDRangeKernel(queue,
			       scan_kern,
			       1,//work_dim,
			       NULL, //global_work_offset
			       global_work_size, //global_work_size
			       local_work_size, //local_work_size
			       0, //num_events_in_wait_list
			       NULL, //event_wait_list
			       NULL //
			       );
  CHK_ERR(err);

  // if further recursive calls are required
  if(left_over > 1)
  {
    // We create a new buffer for the next recursive iteration
    cl_mem g_bbscan = clCreateBuffer(context,CL_MEM_READ_WRITE, 
			      sizeof(int)*left_over,NULL,&err);

    /* Recursively perform scan if needed */
    histogram_helper(queue,context,scan_kern,update_kern,g_bscan,
	     g_bbscan,-1,k,left_over);

    // Upon calculating the 'partial scans' of local chunks, we 
    // set the following arguments for the update kernel.
    err = clSetKernelArg(update_kern,0,
		   sizeof(cl_mem), &out);
    CHK_ERR(err);
    
    err = clSetKernelArg(update_kern,1,
		   sizeof(cl_mem), &g_bbscan);
    CHK_ERR(err);

    err = clSetKernelArg(update_kern,2,
		   sizeof(int), &len);
    CHK_ERR(err);
    
    /* Update partial scans */
    err = clEnqueueNDRangeKernel(queue,
			   update_kern,
			   1,//work_dim,
			   NULL, //global_work_offset
			   global_work_size, //global_work_size
			   local_work_size, //local_work_size
			   0, //num_events_in_wait_list
			   NULL, //event_wait_list
			   NULL //
			   );
    CHK_ERR(err);

    // Release memory associated with temporarily created cl_mem object
    clReleaseMemObject(g_bbscan);
  }

  // Release memory associated with cl_mem object
  clReleaseMemObject(g_bscan);
}
