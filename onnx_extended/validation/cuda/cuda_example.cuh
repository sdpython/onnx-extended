namespace cuda_example {

unsigned int nextPow2(unsigned int x);

void vector_add(unsigned int size, const float *ptr1, const float *ptr2, float *ptr3,
                int cudaDevice);

float vector_sum0(unsigned int size, const float *ptr, int max_threads, int cudaDevice);

float vector_sum_atomic(unsigned int size, const float *ptr, int maxThreads, int cudaDevice);

} // namespace cuda_example
