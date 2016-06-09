#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.cuh"

using namespace std;

/*
 * Mapping function to be run for each input. The input must be read from memory
 * and the the key/value output must be stored in memory at pairs. Multiple
 * pairs may be stored at the next postiion in pairs, but the maximum number of
 * key/value pairs stored must not exceed NUM_KEYS.
 */
__device__ void mapper(input_type *input, KeyValuePair *pairs) {
    int binSize = RAND_MAX / NUM_OUTPUT;
    int binStart = *input - (*input % binSize);
    int binEnd = binStart + binSize;

    pairs->key.start = binStart;
    pairs->key.end = binEnd;
}

/*
 * Reducing function to be run for each set of key/value pairs that share the
 * same key. len key/value pairs may be read from memory, and the output
 * generated from these pairs must be stored at output in memory.
 */
__device__ void reducer(KeyValuePair *pairs, int len, output_type *output) {
    Bin bin = pairs->key;

    output->bin = bin;
    output->count = len;
}

/*
 * Main function that runs a map reduce job.
 */
int main(int argc, char const *argv[]) {
    // Seed the random function for random number generation
    srand (time(NULL));

    // Allocate host memory
    size_t input_size = NUM_INPUT * sizeof(input_type);
    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    input_type *input = (input_type *) malloc(input_size);
    output_type *output = (output_type *) malloc(output_size);

    // Populate the input array with random coordinates
    printf("Generating %d Test Values\n", NUM_INPUT);
    for (size_t i = 0; i < NUM_INPUT; i++) {
        input[i] = rand();
    }

    // Run the Map Reduce Job
    runMapReduce(input, output);

    // Iterate through the output array
    for (size_t i = 0; i < NUM_OUTPUT; i++) {
        printf("Bin: [%d, %d) Count: %d\n", output[i].bin.start, output[i].bin.end, output[i].count);
    }

    // Free host memory
    free(input);
    free(output);

    return 0;
}
