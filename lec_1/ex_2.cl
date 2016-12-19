__kernel void hello(__global double *a, __global double *b,  __global double *c)
{
 size_t id = get_global_id(0);
 c[id] = a[id] + b[id];
}

