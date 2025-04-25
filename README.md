# CUDA_Assignment2_KNS

## Grid and Block Dimensions and Their Responsibilities
For the purposes of this matrix-vector multiplication, a grid of 10 blocks was selected, each of which would compute one element of the result vector.
In each block, I opted for 16 threads each (due to it being a power of two) though only 10 are really used in the computation
Within each block, each thread computed its own partial sum and stored it to shared memory within its block. Then, a parallel summation method was performed to get the final value for the result,
and then stored in the result vector.

## Expected Output
The expected result matrix should contain the same number in each of its 10 entries, the sum of squares of the numbers 1-10 which is 385.

## Compilation and execution
For the purposes of this project, one may simply paste the CUDA file into LeetGPU and execute on their free cloud service.
