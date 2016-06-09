# Cuda MapReduce Implementation
#### by Gavy Aggarwal

<hr>

### Inspiration

After seeing the popularity of MapReduce framework used for distributed computation of large data sets, I implemented it a GPU accelerated version in Cuda.

### How It Works?

The core of the program lies in the `map_reduce.cu` file, which contains the generic code that performs the MapReduce job including the Cuda kernels and key/value pair partitioning. This file should stay the same for all MapReduce jobs.

The `config.cuh` file contains the type definitions and constants necessary for the job. These are specific for each MapReduce job and must be modified by a developer (as in the examples).

In addition, users must supply a definition of a `mapper` and `reducer` function, which transform the input data into a set of key/value pairs and aggregate a set of key/value pairs with the same key into an output respectively.

### Examples

I have provided two examples of MapReduce programs that use my MapReduce implementation. Note that each of these examples could be implemented directly in Cuda to obtain a more optimized version of the program, but that requires Cuda programming experience and dealing with GPU programming. With my MapReduce implementation, a user simply needs to supply the `mapper` and `reducer` functions and identify the types. Then, all of the GPU-accelerated parallel computation runs "under the hood" and doesn't require any explicit programming. I personally spent less than an hour programming each of the two examples below by using the MapReduce implementation that would otherwise take over a day if I were to write and debug the cuda code directly.

#### Uniform Random

This example tests the uniformity of the C++ `rand` function by generating `NUM_INPUT` random integers and placing them in `NUM_OUTPUT` bins. Here, the mapper function is simply placing each input into a correct bin, and the reducer is counting the number of elements in that bin. The output from the program is:

```
Generating 10000000 Test Values
Bin: [0, 214748364) Count: 999913
Bin: [1932735276, 2147483640) Count: 1000605
Bin: [858993456, 1073741820) Count: 998994
Bin: [1717986912, 1932735276) Count: 999659
Bin: [644245092, 858993456) Count: 999163
Bin: [1503238548, 1717986912) Count: 999442
Bin: [429496728, 644245092) Count: 1000233
Bin: [1288490184, 1503238548) Count: 1002615
Bin: [214748364, 429496728) Count: 999478
Bin: [1073741820, 1288490184) Count: 999898
```

Since the counts are approximately equal, we can conclude that the `rand` function is uniform.

#### Pi Estimation

This example tests uses the Monte Carlo method to estimate the value of Pi. `NUM_INPUT` coordinates in the unit square are randomly generated and are tested whether they fit into the unit circle. By using the proportion of points that do and geometry, the value of Pi can be approximated. The output from the program is:

```
Generating 10000000 Test Points
Value of Pi: 3.140716
```

### Optimizations

By nature of the MapReduce model, the Mapper and Reducer have disjoint inputs and require similar computation for many inputs, so they are suitable for parallelization. Thus, instead of running the Mapper on each of inputs in serial as would happen on a single CPU machine, we can now assign an input to each core of a GPU and have thousands of cores run the Mapper simultaneously on different inputs. The Reducer works the same way, but instead of taking in an input, it takes in a range of key/value pairs that share the same key.
