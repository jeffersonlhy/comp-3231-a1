#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

const int FILTER_WIDTH = 3;

//We will only use this filter in part 1
int FILTER[FILTER_WIDTH*FILTER_WIDTH] = {
    0, -1, 0, 
    -1, 5, -1, 
    0, -1, 0
};

// Display the first and last 10 items
// For debug only
void displayResult(const int original[], const int result[], int size) {
	cout << "Display result: ";

	for (int i = 0; i < 10; i++) {
		cout << result[i] << "\n";
	}
	cout << ".\n.\n.\n";

	for (int i = size - 10; i < size; i++) {
		cout << result[i] << "\n";
	}
}

void initData(string file, int **data, int *sizeX, int *sizeY) {
	int x;
	int y;
	long long i = 0;
	cout << "Reading "<< file << "... \n";
	ifstream myfile(file);
	if (myfile.is_open()) {
		myfile >> x;
		myfile >> y;

		int *temp = new int[x * y];
		for( i=0; i < x * y; i++){
			myfile >> temp[(int)i];
		}
		myfile.close();
		*data = temp;
		*sizeX = x;
		*sizeY = y;
	}
	else {
		cout << "ERROR: File " << file << " not found!\n";
		exit(0);
	}
	cout << i << " entries imported\n";
}

// Don't change this code
// We will evaluate your correctness based on the saved result, not printed output
void saveResult(string file, int data[], int sizeX, int sizeY) {
	long long i = 0;
	cout << "Saving data to "<< file <<"... \n";
	ofstream myfile(file, std::ofstream::out);
	if (myfile.is_open()) {
		myfile << sizeX << "\n";
		myfile << sizeY << "\n";
		for (i = 0; i < sizeX * sizeY; i++){
			myfile << data[i] << "\n";
		}
		myfile.close();
	}
	else {
		cout << "ERROR: Cannot save to " << file << "!\n";
		exit(0);
	}
	cout << i << " entries saved\n";
}

//TODO: Implement the kernel function
__global__ void sharpen(int data[], int result[], int sizeX, int sizeY) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < sizeX * sizeY){
		int row = floor(idx / sizeX);
		int col = idx % sizeX;
		int h = 0;
		int tl = (row - 1 > 0 && col - 1 > 0)     		? data[idx - sizeX - 1] 	: 0;
		int tp = (row - 1 > 0) 				      		? data[idx - sizeX] 		: 0;
		int tr = (row - 1 > 0 && col + 1 < sizeX) 		? data[idx - sizeX + 1]		: 0;
		int cl = (col - 1 > 0) 					  		? data[idx - 1] 			: 0;
		int c  = 										  data[idx];
		int cr = (col + 1 < sizeX) 				  		? data[idx + 1] 			: 0;
		int bl = (row + 1 < sizeY && col - 1 > 0) 		? data[idx + sizeX - 1]		: 0;
		int bm = (row + 1 < sizeY) 				  		? data[idx + sizeX]			: 0;
		int br = (row + 1 < sizeY && col + 1 < sizeX)	? data[idx + sizeX]			: 0;

		int source[FILTER_WIDTH * FILTER_WIDTH] = {tl, tp, tr, cl, c, cr, bl, bm, br};
		for (int i = 0; i < FILTER_WIDTH * FILTER_WIDTH; ++i) {
			h += FILTER[i] * source[i];
		}
		result[idx] = h;
	}
}

// GPU implementation
void GPU_Test(int data[], int result[], int sizeX, int sizeY) {
	// input:
	//	int data[] - int array holding the flattened original image
	//	int sizeX - the width of the image
	//	int sizeY - the height of the image
	// output:
	//	int result[] - int array holding the output image
	// TODO: malloc memory, copy input
	int *d_data;
	int *d_result;
	int const size = sizeof(int) * sizeX * sizeY;
	cudaMalloc((void **)&d_data, size);
	cudaMalloc((void **)&d_result, size);
	
	cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, result, size, cudaMemcpyHostToDevice);
	// Start timer for kernel
	// Don't change this function
	int n_blocks = sizeY;
	int BLOCK_SIZE = sizeX;

	auto startKernel = chrono::steady_clock::now();
	// TODO: call the kernel function
	sharpen<<<n_blocks, BLOCK_SIZE>>>(d_data, d_result, sizeX, sizeY);
	// End timer for kernel and display kernel time
	cudaDeviceSynchronize(); // <- DO NOT REMOVE
	auto endKernel = chrono::steady_clock::now();
	cout << "Kernel Elapsed time: " << chrono::duration <double, milli>(endKernel - startKernel).count() << "ms\n";

	// TODO: copy reuslt from device to host
	cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
	// TODO: free device memory <- important, keep your code clean
	cudaFree(d_data); cudaFree(d_result);
}


// CPU implementation
void CPU_Test(int data[], int result[], int sizeX, int sizeY) {
	// input:
	//	int data[] - int array holding the flattened original image
	//	int sizeX - the width of the image
	//	int sizeY - the height of the image
	// output:
	//	int result[] - int array holding the output image

	// TODO: sharpen the image with filter
	//       apply zero padding for the border
	for (int row = 0; row < sizeY; ++row) {
		int row_base = row * sizeX;
		for (int col = 0; col < sizeX; ++col) {
			int h = 0;
			int const idx = row_base + col;
			int tl = (row - 1 > 0 && col - 1 > 0)			? data[idx - sizeX - 1] 	: 0;
			int tp = (row - 1 > 0)							? data[idx - sizeX] 		: 0;
			int tr = (row - 1 > 0 && col + 1 < sizeY)		? data[idx - sizeX + 1]		: 0;
			int cl = (col - 1 > 0)							? data[idx - 1] 			: 0;
			int c  = 										  data[idx];
			int cr = (col + 1 < sizeY) 				  		? data[idx + 1] 			: 0;
			int bl = (row + 1 < sizeX && col - 1 > 0) 		? data[idx + sizeX - 1]		: 0;
			int bm = (row + 1 < sizeX) 				  		? data[idx + sizeX]			: 0;
			int br = (row + 1 < sizeX && col + 1 < sizeY)	? data[idx + sizeX]			: 0;

			int source[FILTER_WIDTH * FILTER_WIDTH] = {tl, tp, tr, cl, c, cr, bl, bm, br};
			for (int i = 0; i < FILTER_WIDTH * FILTER_WIDTH; ++i) {
				h += FILTER[i] * source[i];
			}
			*result = h;
			result++;
		}
	}
	

}

// The input is a 2D grayscale image
// The image is flattened into a text file of pixel values.
int main(int argc, char *argv[]) {
	string inputFile = (argc == 1) ? "image_grey.txt" : argv[1];

	int sizeX;
	int sizeY;
	int *dataForCPUTest;
	int *dataForGPUTest;	

	initData(inputFile, &dataForCPUTest, &sizeX, &sizeY);
	initData(inputFile, &dataForGPUTest, &sizeX, &sizeY);

	int size = sizeX * sizeY;
	int *resultForCPUTest = new int[size];
	int *resultForGPUTest = new int[size];

	cout << "\n";

	cout << "CPU Implementation\n";

	auto startCPU = chrono::steady_clock::now();
	CPU_Test(dataForCPUTest, resultForCPUTest, sizeX, sizeY);
	auto endCPU = chrono::steady_clock::now();

	cout << "Elapsed time: " << chrono::duration <double, milli>(endCPU - startCPU).count() << "ms\n";
	// For debug
	// displayResult(dataForCPUTest, resultForCPUTest, size);

	saveResult("grey_result_CPU.txt",resultForCPUTest, sizeX, sizeY);

	cout << "\n";
	cout << "GPU Implementation\n";

	auto startGPU = chrono::steady_clock::now();
	GPU_Test(dataForGPUTest, resultForGPUTest, sizeX, sizeY);
	auto endGPU = chrono::steady_clock::now();

	cout << "Elapsed time: " << chrono::duration <double, milli>(endGPU - startGPU).count() << "ms\n";

	// For debug
	// displayResult(dataForGPUTest, resultForGPUTest, size);
	saveResult("grey_result_GPU.txt",resultForGPUTest, sizeX, sizeY);

	return 0;
}
