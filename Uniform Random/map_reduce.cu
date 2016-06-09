#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "config.cuh"

using namespace std;

extern __device__ void mapper(input_type *input, KeyValuePair *pairs);
extern __device__ void reducer(KeyValuePair *pairs, int len, output_type *output);

/*
 * Macro to check for GPU errors
 */
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/*
 * An operator definition that allows comparisons between two KeyValuePairs
 * using the StrictWeakOrdering binary predicate. This does a byte by byte
 * comparison of the key, and returns True if the first pair has a key less than
 * second pair.
 */
struct keyValueCompare {
    __host__ __device__ bool operator() (const KeyValuePair &lhs, const KeyValuePair &rhs) {
        void *char_lhs = (unsigned char *) &(lhs.key);
        void *char_rhs = (unsigned char *) &(rhs.key);
        for (int i = 0; i < sizeof(key_type); i++) {
            unsigned char *p1 = (unsigned char *) char_lhs + i;
            unsigned char *p2 = (unsigned char *) char_rhs + i;
            if (*p1 < *p2) {
                return true;
            } else if (*p1 > *p2) {
                return false;
            }
        }
        return false;
    }
};

// Declare mapper and reducer functions
void cudaMap(input_type *input, KeyValuePair *pairs);
void cudaReduce(KeyValuePair *pairs, output_type *output);

/*
 * Mapping Kernel: Since each mapper runs independently of each other, we can
 * give each thread its own input to process and a disjoint space where it can`
 * store the key/value pairs it produces.
 */
__global__ void mapKernel(input_type *input, KeyValuePair *pairs) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < NUM_INPUT;
        i += blockDim.x * gridDim.x) {
        mapper(&input[i], &pairs[i * NUM_KEYS]);
    }
}

/*
 * Reducing Kernel: Given a sorted array of keys, find the range corresponding
 * to each thread and run the reducer on that set of key/value pairs.
 */
__global__ void reduceKernel(KeyValuePair *pairs, output_type *output) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < NUM_OUTPUT;
        i += blockDim.x * gridDim.x) {
        int startIndex = 0;
        int count = 0;
        int valueSize = 0;
        int j;

        for (j = 1; j < NUM_INPUT * NUM_KEYS; j++) {
            if (keyValueCompare()(pairs[j - 1], pairs[j])) {
                if (count == i) {
                    // This thread has found the correct number
                    // There is a bit of warp divergence here as some threads
                    // break before others, but we still make the most out of it
                    // by calling the reducer at the very end, so there is not
                    // any warp divergence where the bulk of the computation
                    // should occur (the reducer).
                    break;
                } else {
                    count++;
                    startIndex = j;
                }
            }
        }

        if (count < i) {
            // This thread doesn't need to process a key. We won't get here, but
            // this code is just there for assurance.
            return;
        }

        valueSize = j - startIndex;

        // Run the reducer
        reducer(pairs + startIndex, valueSize, &output[i]);
    }
}

/*
 * The main function that runs the bulk of the MapReduce job. Space is allocated
 * on the GPU, inputs are copied. The mapper is run. The key/value pairs are
 * sorted. The reducer is run. Output data is copied back from the GPU and
 * returned.
 */
void runMapReduce(input_type *input, output_type *output) {
    // Create device pointers
    input_type   *dev_input;
    output_type  *dev_output;
    KeyValuePair *dev_pairs;

    // Determine sizes in bytes
    size_t input_size = NUM_INPUT * sizeof(input_type);
    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    size_t pairs_size = NUM_INPUT * NUM_KEYS * sizeof(KeyValuePair);

    // Initialize device memory (we can utilize more space by waiting to
    // initialize the output array until we're done with the input array)
    cudaMalloc(&dev_input, input_size);
    cudaMalloc(&dev_pairs, pairs_size);

    // Copy input data over
    cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);
    //cudaMemset(dev_pairs, 0, pairs_size);

    // Run the mapper kernel
    cudaMap(dev_input, dev_pairs);

    // Convert the pointer to device memory for the key/value pairs that is
    // recognizable by the cuda thrust library
    thrust::device_ptr<KeyValuePair> dev_ptr(dev_pairs);

    // Sort the key/value pairs. By using the thrust library, we don't have to
    // write this code ourselves, and it's already optimized for parallel
    // computation
    thrust::sort(dev_ptr, dev_ptr + NUM_INPUT * NUM_KEYS, keyValueCompare());

    // Free GPU space for the input
    cudaFree(dev_input);
    // Allocate GPU space for the output
    cudaMalloc(&dev_output, output_size);

    // Run the reducer kernel
    cudaReduce(dev_pairs, dev_output);

    // Allocate space on the host for the output array and copy the data to it
    cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);

    // Free GPU memory for the key/value pairs and output array
    cudaFree(dev_pairs);
    cudaFree(dev_output);
}

/*
 * Function to call the cuda map kernel and ensure no errors occur
 */
void cudaMap(input_type *input, KeyValuePair *pairs) {
    mapKernel<<<GRID_SIZE, BLOCK_SIZE>>>(input, pairs);
    gpuErrChk( cudaPeekAtLastError() );
    gpuErrChk( cudaDeviceSynchronize() );
}

/*
 * Function to call the cuda reduce kernel and ensure no errors occur
 */
void cudaReduce(KeyValuePair *pairs, output_type *output) {
    reduceKernel<<<GRID_SIZE, BLOCK_SIZE>>>(pairs, output);
    gpuErrChk( cudaPeekAtLastError() );
    gpuErrChk( cudaDeviceSynchronize() );
}
