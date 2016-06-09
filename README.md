# Cuda MapReduce Implementation
#### by Gavy Aggarwal

### Inspiration

MapReduce is a programming model that is typically used to process large datasets on distributed clusters and is popular among large data driven companies such as Google. Today, it’s popular in the form of Hadoop, which is a Java implementation of the programming model. In MapReduce, the developer typically provides a mapper, which is a procedure that takes in a subset of the entire input data and outputs key/value pairs. Then, all the values that share the same key are aggregated and fed to another developer supplied procedure called the reducer, which produces the final output. I decided to use this model to create a framework that runs a MapReduce job on the GPU.

### How It Works?

The core of the program lies in the `map_reduce.cu` file, which contains the generic code that performs the MapReduce job including the Cuda kernels and key/value pair partitioning. This file should stay the same for all MapReduce jobs.

The `config.cuh` file contains the type definitions and constants necessary for the job. These are specific for each MapReduce job and must be modified by a developer (as in the examples).

In addition, users must supply a definition of a `mapper` and `reducer` function, which transform the input data into a set of key/value pairs and aggregate a set of key/value pairs with the same key into an output respectively.

### Examples

I have provided two examples of MapReduce programs that use my MapReduce implementation. Note that each of these examples could be implemented directly in Cuda to obtain a more optimized version of the program, but that requires Cuda programming experience and dealing with GPU programming. With my MapReduce implementation, a user simply needs to supply the `mapper` and `reducer` functions and identify the types. Then, all of the GPU-accelerated parallel computation runs "under the hood" and doesn't require any explicit programming. I personally spent less than an hour programming each of the two examples below by using the MapReduce implementation that would otherwise take over a day if I were to write and debug the cuda code directly.

To run the examples, simply `cd` into the directory for each of the samples and run `make run`.

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

### Performance

I don't think it's suitable to compare this implementation to other MapReduce frameworks because each MapReduce methodology faces different bottlenecks and works differently. For example, Hadoop MapReduce programs are written in Java and parallelization is achieved through distributed computing. Thus, Hadoop's main bottleneck comes from network transfer and synchronization. However, this implementation is written in C++ and doesn't have network transfer. Thus, this would run much faster as C++ doesn't require a JVM and isn't subject to the large network latency and network transfer speeds.

The `Uniform Random` example from above ran in 13.036s. The `Pi Estimation` example from above ran in 3.466s. Running this solely on the CPU takes 29.435s and 5.435s respectively. The GPU version is clearly faster, but again, I would take these numbers with a grain of salt as the speeds vary significantly on the type of CPU/GPU and I suspect a lot of the time for the GPU version comes from IO which will have less of an affect relative to the CPU version with larger data sets.

Also, the speed varies significantly on the user's `Mapper` and `Reducer` function. In the two examples I gave, both of these functions are very simple, so the benefits of parallelization are small compared to when these functions require heavy computation.

### Optimizations

By nature of the MapReduce model, the Mapper and Reducer have disjoint inputs and require similar computation for many inputs, so they are suitable for parallelization. Thus, instead of running the Mapper on each of inputs in serial as would happen on a single CPU machine, we can now assign an input to each core of a GPU and have thousands of cores run the Mapper simultaneously on different inputs. The Reducer works the same way, but instead of taking in an input, it takes in a range of key/value pairs that share the same key.

A less obvious candidate for parallelization is the partitioning procedure that runs between the Mapper and Reducer which aggregates all the key/value pairs produced by the mapper that share the same key so that the reducer can process all the values. To partition this, I sort the list of key/value pairs using Nvidia's `thrust` library which performs common algorithms in parallel. I use a custom comparison operation that greedily compares the bytes of the keys to determine equality.

I also switched from the dynamic memory model to the fixed memory model for further speed improvements. Originally, I implemented this using a dynamic memory model where there could be a variable number of inputs, key/value pairs, and outputs, each with a variable size. However, this significantly slowed down the program due to frequent uncoalesced memory accesses and atomic operations to add new key/value pairs which essentially caused the program to run in serial. Thus, by using a fixed size for the input, output, key, and value types and setting bounds on the number of inputs, outputs, and key/value pairs per input, I was able to perform coalesced memory accesses and eliminating atomic operations by creating disjoint memory regions to write to for each thread. While this poses some constraints on the types of MapReduce jobs possible, I think the speed improvement from a fixed memory model makes this GPU-accelerated MapReduce implementation more viable. Additionally, using the fixed memory model reduces much of the overhead required by the dynamic memory model when each of the inputs, keys, values, and outputs are small in size.

### Limitations & Further Work

The primary limitations in this MapReduce implementation come from memory constraints. As mentioned before, bounds are introduced for the number of keys per input. Thus, the same amount of memory is allocated to store the key/value pairs for each input and if a mapper doesn't produce as many pairs, that memory is wasted. Also, dynamic size variables are not supported, so each element will take up as much memory as the largest element. Thus, it's easy to reach the global memory capacity of the GPU, and only so much data can be processed such that all the key/value pairs fit in the GPU global memory.

This contradicts the goal of MapReduce, to process very large data sets, if we're constrained to the size of GPU global memory, usually a few GB. To get around this, we can design the program as:

```
UNTIL ALL INPUT DATA IS PROCESSED:
    LOAD CHUNK OF INPUT DATA FROM DISK TO GPU
    RUN MAPPER AND STORE PAIRS ON DISK

GROUP ALL THE PAIRS ON DISK USING GPU

UNTIL ALL GROUPS OF PAIRS IS PROCESSED:
    LOAD CHUNK OF GROUPS FROM DISK TO GPU
    RUN REDUCER AND STORE OUTPUT ON DISK
```
By using the disk to store data, we'd be able to process much larger data sets, but using the GPU to efficiently sort a large amount of data on a disk will require lots of complex IO between the disk, host memory, and GPU and is outside the scope of this project.

Another limitation is that this implementation only uses 1 GPU. Greater speed can be achieved by using more GPUs if available, and a similar model as above can be used. This will also require lots of IO and synchronization in the partitioning step and is outside the scope of this project.
