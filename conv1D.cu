#include <iostream>
#include <vector>


__global__ void conv1D(int* mat_dev, int* output_dev, const int size_mat);
void initialize(std::vector<int> &mat_host, std::vector<int> &kernel_host, 
                std::vector<int> &output_host, std::vector<int> &expected_output_host);
bool checkResult(std::vector<int> &output_host, std::vector<int> &expected_output_host);

const int KERNEL_SIZE = 7;

__constant__ int kernel_dev[KERNEL_SIZE];


int main(){

    // General constants
    const int num_iterations = 10000;

    const int num_elements_mat = 2 << 20;

    const int num_bytes_mat = num_elements_mat * sizeof(int);
    const int num_bytes_kernel = KERNEL_SIZE * sizeof(int);

    const dim3 num_threads(64);
    const dim3 num_blocks((num_elements_mat + num_threads.x - 1) / num_threads.x);

    // Create host memory
    std::vector<int> mat_host(num_elements_mat);
    std::vector<int> kernel_host(KERNEL_SIZE);
    std::vector<int> output_host(num_elements_mat);
    std::vector<int> expected_output_host(num_elements_mat);

    initialize(mat_host, kernel_host, output_host, expected_output_host);

    // Create device memory
    int* mat_dev = nullptr;
    int* output_dev = nullptr;

    cudaMalloc(&mat_dev, num_bytes_mat);
    cudaMalloc(&output_dev, num_bytes_mat);

    // Copy data to device memory
    cudaMemcpy(mat_dev, &mat_host[0], num_bytes_mat, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel_dev, &kernel_host[0], num_bytes_kernel, 0, cudaMemcpyHostToDevice);
    cudaMemcpy(output_dev, &output_host[0], num_bytes_mat, cudaMemcpyHostToDevice);

    // Call the kernel
    std::cout << "Launching (" << num_blocks.x << ") blocks and (" << num_threads.x << ") threads for a ";
    std::cout << num_elements_mat << " vector" << std::endl;
    conv1D<<<num_blocks, num_threads>>>(mat_dev, output_dev, num_elements_mat);

    cudaDeviceSynchronize();
    std::cout << "Last CUDA error: " << cudaGetLastError() << " - " << cudaGetErrorString(cudaGetLastError());
    std::cout << std::endl;

    // Copy the result to host
    cudaMemcpy(&output_host[0], output_dev, num_bytes_mat, cudaMemcpyDeviceToHost);

    // Check the result
    bool resultIsCorrect = checkResult(output_host, expected_output_host);

    // Several calls to measure the performance
    for(int it = 0; it < num_iterations; it++){
        conv1D<<<num_blocks, num_threads>>>(mat_dev, output_dev, num_elements_mat);
    }

    // Release memory
    cudaFree(mat_dev);
    cudaFree(output_dev);

    // Return error code base on result checking
    if (resultIsCorrect){
        std::cout << "Result is correct" << std::endl;
        return 0;
    }else{
        std::cout << "Result is wrong" << std::endl;
        return -1;
    }

}


__global__ void conv1D(int* mat_dev, int* output_dev, const int size_mat){

    // Range check
    const int radius = KERNEL_SIZE / 2;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= radius && tid < (size_mat - radius)){
        int temp = 0;

        #pragma unroll
        for(int k = -radius; k < radius; k++){
            temp += mat_dev[tid + k]  * kernel_dev[radius + k];
        }
        output_dev[tid] = temp;
    }

}


void initialize(std::vector<int> &mat_host, std::vector<int> &kernel_host, 
                std::vector<int> &output_host, std::vector<int> &expected_output_host){

    // Default values for mat_host, output_host and expected_output_host
    for(int i = 0; i < mat_host.size(); i++){
        mat_host[i] = i % 3;
        output_host[i] = 0;
        expected_output_host[i] = 0;
    }

    // Default values for kernel_host
    for(int i = 0; i < kernel_host.size(); i++){
        kernel_host[i] = i % 2;
    }

    // Calculate the expected output
    const int radius = kernel_host.size() / 2;
    for(int i = radius; i < (mat_host.size() - radius); i++){  
        for(int k = -radius; k < radius; k++){
            expected_output_host[i] += mat_host[i + k]  * kernel_host[radius + k];
        }
    }

}


bool checkResult(std::vector<int> &output_host, std::vector<int> &expected_output_host){
    return output_host == expected_output_host;
}