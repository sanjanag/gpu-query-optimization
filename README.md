# README #



### What is this repository for? ###

* This repository aims at optimising matrix operations used in optimised graph queries. 
* It exploits the parallelisation offered by GPUs in matrix operations.

### How do I get set up? ###

* Curently two operations have been implemented: fold and unfold
* Configuration
	* Fold - Here, the matrix is split into smaller matrices based on rows ie an mxn matrix wil be split into 4xn matrices if the value of the 'split' variable is 4.
	* Unfold - Here, for each cell in the input matrix there is a separate thread allotted. The threadId is calculated using the blockId. It exploits the 2-dimensional ids that the threads and blocks can have in CUDA. 
* Dependencies
* Database configuration
* How to run fold operation:
```
nvcc fold.cu
./a.out datafile size_of_matrix split iterations
./a.out data1e8.in 100 4 10
```
* How to run unfold operation:
```
nvcc unfold.cu
./a.out datafile size_of_matrix iterations
./a.out data1e8.in 100 10
```

* Deployment instructions

* In case, it shows 'nvcc not installed' run the following command:
```export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}} - See more at: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions```