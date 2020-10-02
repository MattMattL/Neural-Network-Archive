//	Convolutional neural net in C language
//
//	Includes memory allocation, kernel convolution, pooling, scaling
//	and normalisations
//
//	Made as a header file to support a deeper level processing for future
//	projects.
//

#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DEEP_NET_ERROR 1

#define DEEP_NET_MAXPOOLING 1
#define DEEP_NET_AVGPOOLING 2
#define DEEP_NET_MINPOOLING 3

//void cnnTest(double **arrIn, int row, int col);
double** cnnTwoDimMalloc(int row, int col);
void cnnReleaseMemory(double **arr, int row, int col);

// Kernel Convolution
double** cnnRunConvolution(double **arrIn, int *ptrRow, int *ptrCol, int conv_i, int conv_j, double kernel[conv_i][conv_j]);

// Kernel Pooling
double** cnnRunPoolingByOption(double **arrIn, int *ptrRow, int *ptrCol, int conv_i, int conv_j, int opt);
double** cnnRunMaxPooling(double **arrIn, int *arr_row, int *arr_col, int conv_i, int conv_j);
double** cnnRunAvgPooling(double **arrIn, int *arr_row, int *arr_col, int conv_i, int conv_j);
double** cnnRunMinPooling(double **arrIn, int *arr_row, int *arr_col, int conv_i, int conv_j);

// Preprocessing
void cnnProcessMeanRemoval(double **arrIn, int row, int col);
void cnnProcessScaling(double **arrIn, int row, int col, int lower, int upper);

void cnnProcessL1Norm(double **arrIn, int row, int col);
void cnnProcessL2Norm(double **arrIn, int row, int col);
void cnnProcessLnNorm(double **arrIn, int row, int col, double dim);

void cnnTest(double **arrIn, int row, int col)
{
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			arrIn[i][j] = 12.34;
		}
	}
}

void cnnReturnError(char *message)
{
	printf("[error] %s\n", message);

	exit(1);
}

double** cnnTwoDimMalloc(int row, int col)
{
	// Error-checked two-dimensional memory allocation.

	double **tempArr = (double**)malloc(row * sizeof(double*));

	if(tempArr == NULL)
		cnnReturnError("[error] <cnnTwoDimMalloc> layer 1");

	for(int i=0; i<row; i++)
	{
		tempArr[i] = (double*)malloc(col * sizeof(double));

		if(tempArr[i] == NULL)
			cnnReturnError("[error] <cnnTwoDimMalloc> layer 2");
	}

	return tempArr;
}

void cnnReleaseMemory(double **arr, int row, int col)
{
	for(int i=0; i<row; i++)
		free(arr[i]);

	free(arr);
}

double** cnnRunConvolution(double **arrIn, int *ptrRow, int *ptrCol, int conv_i, int conv_j, double kernel[conv_i][conv_j])
{
	int arr_row = *ptrRow;
	int arr_col = *ptrCol;

	int num_i = arr_row - conv_i + 1;
	int num_j = arr_col - conv_j + 1;

	double **arrReturn = cnnTwoDimMalloc(num_i, num_j);

	double sum;

	// field iteration
	for(int row=0; row<num_i; row++)
	{
		for(int col=0; col<num_j; col++)
		{
			sum = 0.0;

			for(int i=0; i<conv_i; i++)
				for(int j=0; j<conv_j; j++)
					sum += arrIn[row + i][col + j] * kernel[i][j];

			arrReturn[row][col] = sum;
		}
	}

	*ptrRow = num_i;
	*ptrCol = num_j;

	return arrReturn;
}

double** cnnRunPoolingByOption(double **arrIn, int *ptrRow, int *ptrCol, int conv_i, int conv_j, int option)
{
	int arr_row = *ptrRow;
	int arr_col = *ptrCol;

	int num_i = arr_row - conv_i + 1;
	int num_j = arr_col - conv_j + 1;

	double **arrReturn = cnnTwoDimMalloc(num_i, num_j);

	double temp;
	double sum;

	// iterate over the given space
	for(int row=0; row<num_i; row++)
	{
		for(int col=0; col<num_j; col++)
		{
			temp = arrIn[row][col];
			sum = 0;

			// shit the kernel
			for(int i=0; i<conv_i; i++)
			{
				for(int j=0; j<conv_j; j++)
				{
					// pooling options
					switch(option)
					{
						case DEEP_NET_MAXPOOLING:
						{
							if(arrIn[row+i][col+j] > temp)
								temp = arrIn[row+i][col+j];
							break;
						}

						case DEEP_NET_AVGPOOLING:
						{
							sum += arrIn[row+i][col+j];
							break;
						}

						case DEEP_NET_MINPOOLING:
						{
							if(arrIn[row+i][col+j] < temp)
								temp = arrIn[row+i][col+j];
							break;
						}
					}
				}
			}

			// copy output
			if(option == DEEP_NET_AVGPOOLING)
				temp = sum / (conv_i * conv_j);
			
			arrReturn[row][col] = temp;
		}
	}

	*ptrRow = num_i;
	*ptrCol = num_j;

	return arrReturn;
}

double** cnnRunMaxPooling(double **arrIn, int *arr_row, int *arr_col, int conv_i, int conv_j)
{
	return cnnRunPoolingByOption(arrIn, arr_row, arr_col, conv_i, conv_j, DEEP_NET_MAXPOOLING);
}

double** cnnRunAvgPooling(double **arrIn, int *arr_row, int *arr_col, int conv_i, int conv_j)
{
	return cnnRunPoolingByOption(arrIn, arr_row, arr_col, conv_i, conv_j, DEEP_NET_AVGPOOLING);
}

double** cnnRunMinPooling(double **arrIn, int *arr_row, int *arr_col, int conv_i, int conv_j)
{
	return cnnRunPoolingByOption(arrIn, arr_row, arr_col, conv_i, conv_j, DEEP_NET_MINPOOLING);
}

void cnnProcessMeanRemoval(double **arrIn, int row, int col)
{
	// calculate the average
	double sum = 0;
	double avg;

	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			sum += arrIn[i][j];
		}
	}

	avg = sum / (row * col);

	// subtract
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			arrIn[i][j] -= avg;
		}
	}
}

void cnnProcessScaling(double **arrIn, int row, int col, int lower, int upper)
{
	// find max and min
	double max = arrIn[0][0];
	double min = arrIn[0][0];

	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			if(arrIn[i][j] > max)
				max = arrIn[i][j];

			if(arrIn[i][j] < min)
				min = arrIn[i][j];
		}
	}

	// scale according to given boundary
	double factor = (upper - lower) / (max - min);

	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			arrIn[i][j] -= min;
			arrIn[i][j] *= factor;
			arrIn[i][j] += lower;
		}
	}

}

void cnnProcessL1Norm(double **arrIn, int row, int col)
{
	double sum;

	for(int i=0; i<row; i++)
	{
		sum = 0.0;

		for(int j=0; j<col; j++)
			sum += fabs(arrIn[i][j]);

		for(int j=0; j<col; j++)
			arrIn[i][j] /= sum;
	}
}

void cnnProcessL2Norm(double **arrIn, int row, int col)
{
	double sum;

	for(int i=0; i<row; i++)
	{
		sum = 0.0;

		for(int j=0; j<col; j++)
			sum += (arrIn[i][j] * arrIn[i][j]);

		for(int j=0; j<col; j++)
			arrIn[i][j] /= sqrt(sum);
	}
}


void cnnProcessLnNorm(double **arrIn, int row, int col, double dim)
{
	if(dim < 1)
		cnnReturnError("[error] Invalid dimension: <cnnProcessNNorm>");

	double sum;

	for(int i=0; i<row; i++)
	{
		sum = 0.0;

		for(int j=0; j<col; j++)
			sum += pow(fabs(arrIn[i][j]), dim);

		for(int j=0; j<col; j++)
			arrIn[i][j] /= pow(sum, 1.0 / 2);
	}
}
