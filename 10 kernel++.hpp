//	A kernel class. Used in ConvNNet++.hpp
//

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <math.h>
#include <time.h>

#define KERNEL_ERROR 0x1100

class Kernel
{
private:

	double** getTwoDimArr(int, int);

public:

	int rows, cols;
	double **at;

	int setKernelSize(int, int);
	void releaseMemory();
	
};

double** Kernel::getTwoDimArr(int rows, int cols)
{
	int i;
	double **arr;

	arr = (double**)malloc(rows * sizeof(double*));

	if(arr == NULL)
		exit(1);

	for(i=0; i<cols; i++)
	{
		arr[i] = (double*)malloc(cols * sizeof(double));

		if(arr[i] == NULL)
			exit(1);
	}

	return arr;
}

int Kernel::setKernelSize(int rowsIn, int colsIn)
{
	if(rowsIn < 1 || colsIn < 1)
		return KERNEL_ERROR;

	rows = rowsIn;
	cols = colsIn;

	at = getTwoDimArr(rows, cols);

	return !KERNEL_ERROR;
}

void Kernel::releaseMemory()
{
	int i;

	for(i=0; i<rows; i++)
		free(at[i]);

	free(at);
}
