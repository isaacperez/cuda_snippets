#include <iostream>

// Forward declarations
void initialize(float* &A_mat_host, float* &B_mat_host, float* &C_mat_host, const int num_elements_for_each_dim);
bool checkResult(float* A_mat_host, float* B_mat_host, float* C_mat_host, const int num_elements_for_each_dim);


__global__ void matmul(float* A_mat_dev, float* B_mat_dev, float* C_mat_dev, const int num_elements_for_each_dim);

#define TILE_SIZE 32

int main(){

    // Constant values for the kernel
    const int num_elements_for_each_dim = 2 << 10;
    const int num_bytes = num_elements_for_each_dim * num_elements_for_each_dim * sizeof(float);

    const int num_iterations = 1;
    
    const dim3 num_threads(TILE_SIZE, TILE_SIZE);
    const dim3 num_blocks((num_elements_for_each_dim + TILE_SIZE - 1) / TILE_SIZE, 
        (num_elements_for_each_dim + TILE_SIZE - 1) / TILE_SIZE);

    // Host memory allocation and initialization
    float* A_mat_host = nullptr;
    float* B_mat_host = nullptr;
    float* C_mat_host = nullptr;

    initialize(A_mat_host, B_mat_host, C_mat_host, num_elements_for_each_dim);

    // Device memory allocation
    float* A_mat_dev = nullptr;
    float* B_mat_dev = nullptr;
    float* C_mat_dev = nullptr;
   
    cudaMalloc(&A_mat_dev, num_bytes);
    cudaMalloc(&B_mat_dev, num_bytes);
    cudaMalloc(&C_mat_dev, num_bytes);

    // Copy the data to device
    cudaMemcpy(A_mat_dev, A_mat_host, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_mat_dev, B_mat_host, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(C_mat_dev, C_mat_host, num_bytes, cudaMemcpyHostToDevice);

    // Call the kernel several times to measure performance
    std::cout << "Calling the kernel " << num_iterations << " times with ";
    std::cout << "(" << num_blocks.x << "," << num_blocks.y << ") blocks and (";
    std::cout << num_threads.x << "," << num_threads.y << ") threads";
    std::cout << " for a " << num_elements_for_each_dim << "x" << num_elements_for_each_dim << " matrix." << std::endl;

    matmul<<<num_blocks, num_threads>>>(A_mat_dev, B_mat_dev, C_mat_dev, num_elements_for_each_dim);
    cudaDeviceSynchronize();

    std::cout << "Last CUDA error code: " << cudaGetLastError() << " - " << cudaGetErrorString(cudaGetLastError());
    std::cout << std::endl;

    for(int it = 0; it < num_iterations; it++){
         matmul<<<num_blocks, num_threads>>>(A_mat_dev, B_mat_dev, C_mat_dev, num_elements_for_each_dim);
    }

    // Copy the data to host
    cudaMemcpy(C_mat_host, C_mat_dev, num_bytes, cudaMemcpyDeviceToHost);

    // Check the result
    bool resultIsCorrect = checkResult(A_mat_host, B_mat_host, C_mat_host, num_elements_for_each_dim);

    // Release host memory
    delete[] A_mat_host;
    delete[] B_mat_host;
    delete[] C_mat_host;

    // Return code based on result checking
    if(resultIsCorrect){
        std::cout << "Result is correct" << std::endl;
        return 0;
    }else{
        std::cout << "Result is wrong" << std::endl;
        return -1;
    }

}


void initialize(float* &A_mat_host, float* &B_mat_host, float* &C_mat_host, const int num_elements_for_each_dim){

    A_mat_host = new float[num_elements_for_each_dim * num_elements_for_each_dim];
    B_mat_host = new float[num_elements_for_each_dim * num_elements_for_each_dim];
    C_mat_host = new float[num_elements_for_each_dim * num_elements_for_each_dim];

    for(int i = 0; i < num_elements_for_each_dim * num_elements_for_each_dim; i++){
        A_mat_host[i] = i % 3;
        B_mat_host[i] = i % 5;
        C_mat_host[i] = 0.0f;
    }
    
}


bool checkResult(float* A_mat_host, float* B_mat_host, float* C_mat_host, const int num_elements_for_each_dim){

    for(int row = 0; row < num_elements_for_each_dim; row++){
        
        if (row % 10 == 0){
            std::cout << "Checking row " << row << " of " << num_elements_for_each_dim << "...\r";
        }

        for(int col = 0; col < num_elements_for_each_dim; col++){

            // Calculate the expected value for C[row, col] = reduce_sum(A[row, :] * B[:, col])
            float expected_value = 0.0f;
            for(int element = 0; element < num_elements_for_each_dim; element++){
               expected_value += A_mat_host[row * num_elements_for_each_dim + element] * 
                    B_mat_host[element * num_elements_for_each_dim + col];
            }

            // Check the value is correct
             if (C_mat_host[row * num_elements_for_each_dim + col] != expected_value){
                std::cout << "Found " << C_mat_host[row * num_elements_for_each_dim + col];
                std::cout << " in (" << row << "," << col << ") position. Expected: " << expected_value << std::endl;
                return false;
             }


        }
    }
    std::cout << std::endl;

    return true;
}


__global__ void matmul(float* A_mat_dev, float* B_mat_dev, float* C_mat_dev, const int num_elements_for_each_dim){

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    const int base_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int base_col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Move the tile in each dimension
    float result = 0.0f;
    for(int shift = 0; shift < num_elements_for_each_dim; shift+=TILE_SIZE){

        // Copy to shared memory the current tile
        tileA[threadIdx.y][threadIdx.x] = A_mat_dev[base_row * num_elements_for_each_dim + shift + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B_mat_dev[(shift + threadIdx.y) * num_elements_for_each_dim + base_col];

        __syncthreads();

        // Calculate the product in the current tile and sum it up
        for(int element = 0; element < TILE_SIZE; element++){
            result += tileA[threadIdx.y][element] * tileB[element][threadIdx.x];
        }
        __syncthreads();
    }

    // Save the result
    C_mat_dev[base_row * num_elements_for_each_dim + base_col] = result;

}