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
    // We set the key of each input to 0.
    pairs->key = 0;
    float x = input->x;
    float y = input->y;
    // We check if the input point fits within the unit circle, and set the
    // value accordingly
    if (x * x + y * y <= 1) {
        pairs->value = 1;
    } else {
        pairs->value = 0;
    }
}

/*
 * Reducing function to be run for each set of key/value pairs that share the
 * same key. len key/value pairs may be read from memory, and the output
 * generated from these pairs must be stored at output in memory.
 */
__device__ void reducer(KeyValuePair *pairs, int len, output_type *output) {
    //int key = pairs->key;
    // We calculate the proportion of the points within the unit circle
    int pointsIn = 0;
    for (KeyValuePair *pair = pairs; pair != pairs + len; pair++) {
        if(pair->value == 1) {
            pointsIn++;
        }
    }
    // We multiply the proportion by 4, since our points are in only the quarter
    // circle, so by geometry, the value of pi is 4 times the number of points
    // in the unit circle out of the unit square
    *output = 4.0 * (float(pointsIn)/float(len));
}

/*
 * Generate a random float between 0 and 1 inclusive.
 */
float randFloat() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
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
    printf("Generating %d Test Points\n", NUM_INPUT);
    for (size_t i = 0; i < NUM_INPUT; i++) {
        input[i].x = randFloat();
        input[i].y = randFloat();
    }

    // Run the Map Reduce Job
    runMapReduce(input, output);

    // Iterate through the output array
    for (size_t i = 0; i < NUM_OUTPUT; i++) {
        printf("Value of Pi: %f\n", output[i]);
    }

    // Free host memory
    free(input);
    free(output);

    return 0;
}
