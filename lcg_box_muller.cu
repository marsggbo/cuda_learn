#include <iostream>  
#include <cmath>  
  
#define BLOCK_SIZE 256  
#define LCG_A 1664525  
#define LCG_C 1013904223  
#define LCG_M 0xFFFFFFFF


__device__ float lcg_random(unsigned int* state) {  
    *state = (*state) * LCG_A + LCG_C;  
    return static_cast<float>(*state) / LCG_M;  
}  
  
__device__ void box_muller_transform(float u1, float u2, float &z1, float &z2) {  
    float r = sqrtf(-2.0f * logf(u1));  
    float theta = 2.0f * M_PI * u2;  
    z1 = r * cosf(theta);  
    z2 = r * sinf(theta);  
}  
  
__global__ void generate_normal_distribution(float* rand_nums, unsigned int seed, int size) {  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  
    int stride = blockDim.x * gridDim.x;  
  
    unsigned int state = seed + tid;  
  
    for (int i = tid; i < size; i += stride) {  
        float u1 = lcg_random(&state);  
        float u2 = lcg_random(&state);  
        float z1, z2;
        box_muller_transform(u1, u2, z1, z2);  
        rand_nums[i] = z1;  
        if(i + stride < size) rand_nums[i + stride] = z2;  
    }  
}  
  
int main() {  
    const int size = 10; // Tensor大小  
  
    float* d_rand_nums;  
    cudaMalloc((void**)&d_rand_nums, size * sizeof(float));  
  
    unsigned int seed = time(NULL); // 设置种子  
  
    generate_normal_distribution<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_rand_nums, seed, size);  
  
    float* rand_nums = new float[size];  
    cudaMemcpy(rand_nums, d_rand_nums, size * sizeof(float), cudaMemcpyDeviceToHost);  
  
    // 在这里可以将rand_nums转换为Tensor或按需使用 
    for (int i = 0; i < size; ++i)  
    {  
        std::cout << rand_nums[i] << " ";  
    }  
    std::cout << std::endl;   
  
    delete[] rand_nums;  
    cudaFree(d_rand_nums);  
  
    return 0;  
}  