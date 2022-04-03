#include <iostream>
#include <vector>


__global__ void reduceCol(int* input_dev, int* output_dev, const int num_elements_by_dim);
void initialize(std::vector<int> &input_host, std::vector<int> &output_host, std::vector<int> &expected_output_host, 
    const int num_elements_by_dim);
bool checkResult(std::vector<int> &output_host, std::vector<int> &expected_output_host);


int main(){

    // General constants
    const int num_iterations = 1000;

    const int num_elements_by_dim = 1 << 10;
    const int num_elements = num_elements_by_dim * num_elements_by_dim;

    const int num_bytes_by_dim = num_elements_by_dim * sizeof(int);
    const int num_bytes = num_elements * sizeof(int);

    const dim3 num_threads(32);
    const dim3 num_blocks(256);

    // Create host memory
    std::vector<int> input_host(num_elements);
    std::vector<int> output_host(num_elements_by_dim);
    std::vector<int> expected_output_host(num_elements_by_dim);

    initialize(input_host, output_host, expected_output_host, num_elements_by_dim);

    // Create device memory
    int* input_dev = nullptr;
    int* output_dev = nullptr;

    cudaMalloc(&input_dev, num_bytes);
    cudaMalloc(&output_dev, num_bytes_by_dim);

    // Copy data to device
    cudaMemcpy(input_dev, &input_host[0], num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(output_dev, &output_host[0], num_bytes_by_dim, cudaMemcpyHostToDevice);

    // Call the kernel
    std::cout << "Launching (" << num_blocks.x << ") blocks and (" << num_threads.x << ") threads for a ";
    std::cout << num_elements_by_dim << "x" << num_elements_by_dim << " matrix" << std::endl;
    reduceCol<<<num_blocks, num_threads, num_blocks.x>>>(input_dev, output_dev, num_elements_by_dim);

    cudaDeviceSynchronize();
    std::cout << "Last CUDA error: " << cudaGetLastError() << " - " << cudaGetErrorString(cudaGetLastError());
    std::cout << std::endl;

    // Get the result
    cudaMemcpy(&output_host[0], output_dev, num_bytes_by_dim, cudaMemcpyDeviceToHost);

    // Check the result
    bool resultIsCorrect = checkResult(output_host, expected_output_host);

    // Several calls to measure the performance
    for(int it = 0; it < num_iterations; it++){
        reduceCol<<<num_blocks, num_threads, num_blocks.x>>>(input_dev, output_dev, num_elements_by_dim);
    }

    // Release memory
    cudaFree(input_dev);
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


__global__ void reduceCol(int* input_dev, int* output_dev, const int num_elements_by_dim){

    extern __shared__ int temp_sums[];

    // Each block process a row and all its multiples
    for(int row = blockIdx.x; row < num_elements_by_dim; row += gridDim.x){

        // Each thread calculate the sum of all the values of its slice
        int temp_sum = 0;
        for(int col = threadIdx.x; col < num_elements_by_dim; col+=blockDim.x){
            temp_sum += input_dev[row * num_elements_by_dim + col];
        }

        // Save the result in shared memory
        temp_sums[threadIdx.x] = temp_sum;

        // Wait for the result of all threads
        __syncthreads();

        // Apply reduction to shared memory
        for(int t = blockDim.x / 2; t > 0; t >>= 1){
            if(threadIdx.x < t){
                temp_sums[threadIdx.x] += temp_sums[threadIdx.x + t];
            }
            __syncthreads();
        }

        // Save the result to global memory
        if (threadIdx.x == 0){
            output_dev[row] = temp_sums[0];
        }

        // Wait for the write in global memory
        __syncthreads();
  
    }

}


void initialize(std::vector<int> &input_host, std::vector<int> &output_host, std::vector<int> &expected_output_host, 
    const int num_elements_by_dim){

    for(int i = 0; i < num_elements_by_dim; i++){
        int sum = 0;
        for(int j = 0; j < num_elements_by_dim; j++){
            input_host[i * num_elements_by_dim + j] = i % 10;
            sum += input_host[i * num_elements_by_dim + j];
        }
        output_host[i] = 0;
        expected_output_host[i] = sum;
    }

}


bool checkResult(std::vector<int> &output_host, std::vector<int> &expected_output_host){
    return output_host == expected_output_host;
}