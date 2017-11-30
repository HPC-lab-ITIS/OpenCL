__kernel void reduce_simd(__global float* a, __local float* buffer, __global float* result, const int n)
{
  int global_index = get_global_id(0);
  int local_index = get_local_id(0);

  if (global_index < n) 
    buffer[local_index] = a[global_index] + a[global_index + n];

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