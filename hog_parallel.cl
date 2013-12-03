__kernel void compute_histogram(


  ) 
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);

  if (idx < n && gid > 0) {
    
  }
}