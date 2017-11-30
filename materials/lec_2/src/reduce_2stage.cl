__kernel void reduce_stage(__global float* a, __local float* buffer, __global float* result, const int n)
{
  int global_index = get_global_id(0);
  int local_index = get_local_id(0);
  float sum_tmp = 0.;

  while(global_index < n)
  {
      sum_tmp += a[global_index];
      global_index +=  get_global_size(0);
  }

  buffer[local_index] = sum_tmp;

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = get_local_size(0) / 2; offset > 0; offset >>= 1) 
  {
    if (local_index < offset) 
      buffer[local_index] += buffer[local_index + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_index == 0)
    result[get_group_id(0)] = buffer[0];
}