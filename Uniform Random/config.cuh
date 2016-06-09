#ifndef MAP_REDUCE_CUH
#define MAP_REDUCE_CUH

// Configure GPU parameters
#define GRID_SIZE 1024
#define BLOCK_SIZE 1024

// Set number of input elements, number of output elements, and number of keys
// per input element
#define NUM_INPUT 10000000
#define NUM_OUTPUT 10
#define NUM_KEYS 1

// Example of custom key and output type
struct Bin {
    int start;
    int end;
};

struct BinFrequency {
    Bin bin;
    int count;
};

// Setting input, output, key, and value types
typedef int input_type;
typedef BinFrequency output_type;
typedef Bin key_type;
typedef char value_type;

// Do not edit below this line

struct KeyValuePair {
   key_type key;
   value_type value;
};

void runMapReduce(input_type *input, output_type *output);

#endif
