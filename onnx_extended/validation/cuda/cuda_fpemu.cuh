namespace fpemu {

void fpemu_cuda_forward(const int size, const float *input, uint8_t *output,
                        FpemuMode mode, bool inplace, float scale,
                        bool block_norm, int block_size);

} // namespace fpemu