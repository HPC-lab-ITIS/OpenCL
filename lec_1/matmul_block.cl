__kernel void matmul(__global double* a, __global double* b, __global double* c, const int n)
{
    const int block_size = 32;
    //Подматрица A в разделяемой памяти
    __local double a_s[32][32];
    //Подматрица B в разделяемой памяти
    __local double b_s[32][32];

    //Вычисляемый элемент C’
    double sum = 0.;

    size_t i = get_global_id(0);//threadIdx.y + blockDim.y * blockIdx.y;
    size_t j = get_global_id(1);//threadIdx.x + blockDim.x * blockIdx.x;

    //Цикл по подматрицам
    for (size_t m = 0; m < n / block_size; ++m)
    {
        // Загрузить по одному элементу из A и B в разделяемую память
        a_s[get_local_id(0)][get_local_id(1)] = a[i * n + (m * block_size + get_local_id(1))];
        b_s[get_local_id(0)][get_local_id(1)] = b[(m * block_size + get_local_id(0)) * n + j];

        //Дождаться когда обе подматрицы будут полностью загружены
        barrier(CLK_LOCAL_MEM_FENCE);

        //Вычислить элемент произведения загруженных подматриц
        for (size_t k = 0; k < block_size; k++)
            sum += a_s[get_local_id(0)][k] * b_s[k][get_local_id(1)];

        //Дождаться пока все нити блока закончат
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //Сохранить результат в глобальной памяти
    c [i * n + j] = sum;
}