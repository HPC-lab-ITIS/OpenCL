__kernel void reduce_vec(__global float* a, __local float* buffer, __global float* result, const int n)
{
  int global_index = get_global_id(0);
  int local_index = get_local_id(0);
  float16 sum_tmp = (float16)(0.);

  while(global_index < n / 16)
  {
      sum_tmp += ((__global float16*)a)[global_index];
      global_index += get_global_size(0);
  }

  buffer[local_index] = sum_tmp.s0 + sum_tmp.s1 + sum_tmp.s2 + sum_tmp.s3 + sum_tmp.s4
                      + sum_tmp.s5 + sum_tmp.s6 + sum_tmp.s7 + sum_tmp.s8 + sum_tmp.s9
                      + sum_tmp.sa + sum_tmp.sb + sum_tmp.sc + sum_tmp.sd + sum_tmp.se 
                      + sum_tmp.sf;

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