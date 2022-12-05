#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

//We may change this value!!
const int FILTER_WIDTH = 7;

//We may change this value!!!
int FILTER[FILTER_WIDTH*FILTER_WIDTH] = {
	1,4,7,10,7,4,1,
	4,12,26,33,26,12,4,
	7,26,55,71,55,26,7,
	10,33,71,91,71,33,10,
	7,26,55,71,55,26,7,
	4,12,26,33,26,12,4,
	1,4,7,10,7,4,1
};

// Display the first and last 10 items
// For debug only
void displayResult(const int original[], const int result[], int size) {
	cout << "Display result: ";
	cout << "(original -> result)\n";

	for (int i = 0; i < 10; i++) {
		cout << original[i] << " -> " << result[i] << "\n";
	}
	cout << ".\n.\n.\n";

	for (int i = size - 10; i < size; i++) {
		cout << original[i] << " -> " << result[i] << "\n";
	}
}

void initColorData(string file, int **data, int *sizeX, int *sizeY) {
	int x;
	int y;
	long long i = 0;
	cout << "Reading "<< file << "... \n";
	ifstream myfile(file);
	if (myfile.is_open()) {
		myfile >> x;
		myfile >> y;

		int *temp = new int[x * y * 3];
		for( i=0; i < x * y * 3; i++){
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

void saveResult(string file, int data[], int sizeX, int sizeY) {
	long long i = 0;
	cout << "Saving data to "<< file <<"... \n";
	ofstream myfile(file, std::ofstream::out);
	if (myfile.is_open()) {
		myfile << sizeX << "\n";
		myfile << sizeY << "\n";
		for (i = 0; i < sizeX * sizeY; i++){
			myfile << data[3* i] << " " << data[3* i + 1] << " " << data[3* i+ 2]<< "\n";
		}
		myfile.close();
	}
	else {
		cout << "ERROR: Cannot save to " << file << "!\n";
		exit(0);
	}
	cout << i << " entries saved\n";
}

// TODO: implement the kneral function for 2D smoothing 
__global__ void sharpen_3d(int data[], int result[], int sizeX, int sizeY) {
	int idx = (blockIdx.x*blockDim.x + threadIdx.x) * 3;
	if (idx < sizeX * sizeY * 3){
		int row = floor(idx / (sizeX * 3));
		int col = idx % (sizeX * 3);
		int h = 0;
		int filter_center = floor(FILTER_WIDTH / 2);
		for (int i = 0; i < FILTER_WIDTH; ++i) {
			int row_diff = floor(i / FILTER_WIDTH) - filter_center;
			int col_diff = i % FILTER_WIDTH - filter_center;
			int final_row = row + row_diff;
			int final_col = col + col_diff;
			if (final_row >= 0 && final_row < sizeY && 
				final_col >= 0 && final_col < sizeX) {
				int final_idx = sizeX * 3 * row + col * 3;
				h += data[final_idx] * FILTER[i];
				h += data[final_idx + 1] * FILTER[i];
				h += data[final_idx + 2] * FILTER[i];
			}
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
	//	int result[] - int array holding the image
	int *d_data;
	int *d_result;

	// TODO: allocate device memory and copy data onto the device
	cudaMalloc((void **)&d_data, sizeof(int) * sizeX * sizeY * 3);
	cudaMalloc((void **)&d_result, sizeof(int) * sizeX * sizeY * 3);

	cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, result, size, cudaMemcpyHostToDevice);
	// Start timer for kernel
	auto startKernel = chrono::steady_clock::now();

	// TODO: call the kernel function
	sharpen_3d<<<sizeX, sizeY * 3>>>(d_data, d_result, sizeX, sizeY);
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
	//	int result[] - int array holding the image

	// TODO: smooth the image with filter size = FILTER_WIDTH
	//       apply zero padding for the border
	for (int idx = 0; idx < sizeX * sizeY * 3; ++idx) {
		int row = floor(idx / (sizeX * 3));
		int col = idx % (sizeX * 3);
		int h = 0;
		int filter_center = floor(FILTER_WIDTH / 2);
		for (int i = 0; i < FILTER_WIDTH; ++i) {
			int row_diff = floor(i / FILTER_WIDTH) - filter_center;
			int col_diff = i % FILTER_WIDTH - filter_center;
			int final_row = row + row_diff;
			int final_col = col + col_diff;
			if (final_row >= 0 && final_row < sizeY && 
				final_col >= 0 && final_col < sizeX) {
				int final_idx = sizeX * 3 * row + col * 3;
				h += data[final_idx] * FILTER[i];
				h += data[final_idx + 1] * FILTER[i];
				h += data[final_idx + 2] * FILTER[i];
			}
		}
		result[idx] = h;
	};
};

// The image is flattened into a text file of pixel values.
int main(int argc, char *argv[]) {
	string inputFile = (argc == 1) ? "image_color.txt" : argv[1];

	int sizeX;
	int sizeY;
	int *dataForCPUTest;
	int *dataForGPUTest;	

	initColorData(inputFile, &dataForCPUTest, &sizeX, &sizeY);
	initColorData(inputFile, &dataForGPUTest, &sizeX, &sizeY);

	int size = sizeX * sizeY * 3;
	int *resultForCPUTest = new int[size];
	int *resultForGPUTest = new int[size];

	cout << "\n";

	cout << "CPU Implementation\n";

	auto startCPU = chrono::steady_clock::now();
	CPU_Test(dataForCPUTest, resultForCPUTest, sizeX, sizeY);
	auto endCPU = chrono::steady_clock::now();

	// cout << "Elapsed time: " << chrono::duration <double, milli>(endCPU - startCPU).count() << "ms\n";

	// // displayResult(dataForCPUTest, resultForCPUTest, size);

	// saveResult("color_result_CPU.txt",resultForCPUTest, sizeX, sizeY);

	// cout << "\n";
	// cout << "GPU Implementation\n";

	// auto startGPU = chrono::steady_clock::now();
	// GPU_Test(dataForGPUTest, resultForGPUTest, sizeX, sizeY);
	// auto endGPU = chrono::steady_clock::now();

	// cout << "Elapsed time: " << chrono::duration <double, milli>(endGPU - startGPU).count() << "ms\n";

	// // displayResult(dataForGPUTest, resultForGPUTest, size);
	// saveResult("color_result_GPU.txt",resultForGPUTest, sizeX, sizeY);

	return 0;
}
