# General Concepts
- All CUDA kernel calls are __asynchronous__ so we have to call `cudaDeviceSynchronize()` to wait for the result or
copy from the memory, which is a blocking operation. Usually, we do not have to worry about this. We use the default
stream to launch the kernel, which is serial, so, if we use memory copies operations, we will have our data avaiable.

- Kernel call: `<<<num_blocks, num_threads, shared_memory_byte_size, stream>>>`. By default, `shared_memory_byte_size` 
is 0 and `stream' is 0 too, which mean default stream.

- `extern` keyword allows you to use shared memory dynamicly without defining the size: `extern __shared__ float sh[]`.

- If we have `N` elements to process and we want to launch `T` threads and have a thread processing each elements, 
to calculate the number of block we can use the following expression: `int num_blocks = ( N + T - 1) / T)`. This way,
the number of blocks will be always the needed to have one thread for each element or even more in case the number 
of element is not evenly divisible by the number of threads.

- Program compilation: `nvcc file.cu -o bin_name.exe`.
- Program profiling: `nvprof .\bin_name.exe`. Always use a large number of elements and call the kernel thousands of
times to get realistic results. The GPU is not always running at its high level and need a warm-up.

- Basic CUDA program structure:
```cpp
// Forward declaration

int main(){

    // Create host pointers and initialize them

    // Create device pointers

    // Copy the host data to device

    // Call the kernel several times

    // Check last CUDA error code

    // Copy the result to host

    // Check the result is correct

    // Release host memory

    // Release CUDA memory

    // Return error code if the result is not correct
}

// Implementation

```