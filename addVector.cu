#include <iostream>
#include <cuda_runtime.h>


void initialize(double* &vec_a_host, double* &vec_b_host, double* &vec_c_host, const int num_elements);
bool checkResult(double* vec_a_host, double* vec_b_host, double* vec_c_host, const int num_elements);


__global__ void addVector(double* vec_a_dev, double* vec_b_dev, double* vec_c_dev, const int num_elements);


int main(){
    
    // Declare the size of the data
    const int num_elements = (2 << 20) + 1;
    const int num_bytes = num_elements * sizeof(double);

    // Declare the constants of the kernel
    const int num_calls = 20000;
    dim3 threads(32);
    dim3 blocks((num_elements + threads.x - 1) / threads.x);

    // Intialize host data
    double* vec_a_host = nullptr;
    double* vec_b_host = nullptr;
    double* vec_c_host = nullptr;

    initialize(vec_a_host, vec_b_host, vec_c_host, num_elements);

    // Create device data
    double* vec_a_dev = nullptr;
    double* vec_b_dev = nullptr;
    double* vec_c_dev = nullptr;

    cudaMalloc(&vec_a_dev, num_bytes);
    cudaMalloc(&vec_b_dev, num_bytes);
    cudaMalloc(&vec_c_dev, num_bytes);

    // Copy host data to device
    cudaMemcpy(vec_a_dev, vec_a_host, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(vec_b_dev, vec_b_host, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(vec_c_dev, vec_c_host, num_bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    std::cout << "Launching kernel with (" << blocks.x << ") blocks and (" << threads.x << ") threads for ";
    std::cout << num_elements << " elements." << std::endl;

    for(int i = 0; i < num_calls; i++){
        addVector<<<blocks, threads>>>(vec_a_dev, vec_b_dev, vec_c_dev, num_elements);
    }

    std::cout << "Last CUDA error code: " << cudaGetLastError() << std::endl;

    // Copy device data to host
    cudaMemcpy(vec_c_host, vec_c_dev, num_bytes, cudaMemcpyDeviceToHost);

    // Check the result
    bool resultIsCorrect = checkResult(vec_a_host, vec_b_host, vec_c_host, num_elements);

    // Remove host data
    delete[] vec_a_host;
    delete[] vec_b_host;
    delete[] vec_c_host;

    // Remove device data
    cudaFree(vec_a_dev);
    cudaFree(vec_b_dev);
    cudaFree(vec_c_dev);

    // Return code depends of the validation
    if(resultIsCorrect){
        std::cout << "Result is correct" << std::endl;
        return 0;
    }else{
        std::cout << "Result is not correct" << std::endl;
        return -1;
    }

}


void initialize(double* &vec_a_host, double* &vec_b_host, double* &vec_c_host, const int num_elements){

    // Allocate the memory
    vec_a_host = new double[num_elements];
    vec_b_host = new double[num_elements];
    vec_c_host = new double[num_elements];

    // Initialize
    for(int i = 0; i < num_elements; i++){
        vec_a_host[i] = 1;
        vec_b_host[i] = 2;
        vec_c_host[i] = 0;
    }

}


bool checkResult(double* vec_a_host, double* vec_b_host, double* vec_c_host, const int num_elements){

    for(int i = 0; i < num_elements; i++){
        if(vec_a_host[i] + vec_b_host[i] != vec_c_host[i]){
            
            std::cout << "Expected " << vec_a_host[i] + vec_b_host[i] << " in position " << i << ". Found ";
            std::cout << vec_c_host[i] << std::endl;

            return false;
        }
    }

    return true;
}


__global__ void addVector(double* vec_a_dev, double* vec_b_dev, double* vec_c_dev, const int num_elements){

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_elements){
        vec_c_dev[idx] = vec_a_dev[idx] + vec_b_dev[idx];
    }

}