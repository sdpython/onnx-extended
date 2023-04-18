namespace cuda_example {

void vector_add(unsigned int size, const float *ptr1, const float *ptr2,
                float *ptr3, int cudaDevice);

float vector_sum(unsigned int size, const float *ptr, int max_threads, int cudaDevice);

} // namespace cuda_example
