#include <iostream>
#include <vector>
#include <cuda_runtime.h>


__global__ void calcHist(int* matrix_dev, int* hist_dev, const int bin_size, const int num_elements_for_each_dim_mat,
    const int num_bins);
void initialize(std::vector<int> &matrix_host, std::vector<int> &hist_host, std::vector<int> &expected_hist_host,
    const int num_values, const int range_size);
bool checkResult(std::vector<int> &hist_host, std::vector<int> &expected_hist_host);


int main(){

    // General constants
    const int num_iterations = 10;

    const int num_elements_for_each_dim_mat = 1 << 10;
    const int num_elements_mat = num_elements_for_each_dim_mat * num_elements_for_each_dim_mat;
    const int num_bytes_mat = num_elements_mat * sizeof(int);
    
    const int num_elements_bin = 10;
    const int num_bytes_bin = num_elements_bin * sizeof(int);
    
    const int num_values = 100;
    const int range_size = (num_values + num_elements_bin - 1) / num_elements_bin;

    const dim3 num_threads(16, 16);
    const dim3 num_blocks((num_elements_for_each_dim_mat + num_threads.x -1) / num_threads.x, 
        (num_elements_for_each_dim_mat + num_threads.y -1) / num_threads.y);

    // Host memory allocation
    std::vector<int> matrix_host(num_elements_mat);
    std::vector<int> hist_host(num_elements_bin);
    std::vector<int> expected_hist_host(num_elements_bin);

    initialize(matrix_host, hist_host, expected_hist_host, num_values, range_size);

    // Device memory allocation
    int* matrix_dev = nullptr;
    int* hist_dev = nullptr;

    cudaMalloc(&matrix_dev, num_bytes_mat);
    cudaMalloc(&hist_dev, num_bytes_bin);

    // Copy memory to device
    cudaMemcpy(matrix_dev, &matrix_host[0], num_bytes_mat, cudaMemcpyHostToDevice);
    cudaMemcpy(hist_dev, &hist_host[0], num_bytes_bin, cudaMemcpyHostToDevice);

    // Call the kernel
    std::cout << "Launching (" << num_blocks.x << "," << num_blocks.y << ") blocks and (";
    std::cout << num_threads.x << "," << num_threads.y << ") threads for ";
    std::cout << " for a " << num_elements_for_each_dim_mat << "x" << num_elements_for_each_dim_mat << " matrix";
    std::cout << std::endl;

    std::cout << "Num values: " << num_values << ", Num bins: " << num_elements_bin << ", Range size: " << range_size;
    std::cout << std::endl;

    calcHist<<<num_blocks, num_threads, num_elements_bin>>>(
        matrix_dev, hist_dev, range_size, num_elements_for_each_dim_mat, num_elements_bin);

    cudaDeviceSynchronize();
    std::cout << "Last CUDA error: " << cudaGetLastError() << " - " << cudaGetErrorString(cudaGetLastError());
    std::cout << std::endl;

    // Copy the result to host
    cudaMemcpy(&hist_host[0], hist_dev, num_bytes_bin, cudaMemcpyDeviceToHost);

    // Check the result
    bool resultIsCorrect = checkResult(hist_host, expected_hist_host);

    // Performance measure
    for(int it = 0; it < num_iterations; it++){
        calcHist<<<num_blocks, num_threads, num_elements_bin>>>(
            matrix_dev, hist_dev, range_size, num_elements_for_each_dim_mat, num_elements_bin);
    }
   
    // Release memory
    cudaFree(matrix_dev);
    cudaFree(hist_dev);

    if (resultIsCorrect){
        std::cout << "Result is correct" << std::endl;
        return 0;
    }else{
        std::cout << "Result is wrong" << std::endl;
        return -1;
    }

}


__global__ void calcHist(int* matrix_dev, int* hist_dev, const int bin_size, const int num_elements_for_each_dim_mat,
    const int num_bins){

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ int hist_sm[];

    // Prepare shared memory
    if (threadIdx.y * blockDim.x + threadIdx.x < num_bins){
        hist_sm[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    } 
    __syncthreads();

    // Check if we are in a valid place
    if (row < num_elements_for_each_dim_mat && col < num_elements_for_each_dim_mat){
        // There is a thread over each pixel of matrix_dev
        atomicAdd(&hist_sm[matrix_dev[row * num_elements_for_each_dim_mat + col] / bin_size], 1);

    }
    __syncthreads();

    // Copy to global memory
    if (threadIdx.y * blockDim.x + threadIdx.x < num_bins){
        atomicAdd(&hist_dev[threadIdx.y * blockDim.x + threadIdx.x], hist_sm[threadIdx.y * blockDim.x + threadIdx.x]);
    }   
}


void initialize(std::vector<int> &matrix_host, std::vector<int> &hist_host, std::vector<int> &expected_hist_host,
    const int num_values, const int range_size){
    for(int i = 0; i < matrix_host.size(); i++){
        matrix_host[i] = i % num_values;
        expected_hist_host[matrix_host[i] / range_size]++;
    }
}

bool checkResult(std::vector<int> &hist_host, std::vector<int> &expected_hist_host){
    return hist_host == expected_hist_host;
}