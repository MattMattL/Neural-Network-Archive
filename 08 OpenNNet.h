//	Two-layer neural net defined as a header file
//
//	Added more functions for error handling and input/output. Memory
//	protection was also heavily taken into account.
//

#include <stdlib.h>
#include <time.h>
#include <math.h>

#define OPEN_NET_ERROR 1

const char directory[64] = "./neural_weights";

int NUM_IN = -1, NUM_MID = -1, NUM_OUT = -1;

//deprecated
int FILE_ID = -99999;

double *productSum_L1, *productSum_L2;
double *sigmoidOut_L1, *sigmoidOut_L2;

double **weight_L1, **weight_L2;
double *delta1, *delta2;

void nnBarf(char message[]);
int nnReturnNormal();
int nnReturnError(char string[]);

int nnPrintWeightInfo(int layer);
double nnGetSingleWeightInfo(int layer, int percA, int percB);
double nnGetSingleNetResult(int numOut);

void nnPrintTest();
int nnSetNetworkSize(int num_in, int num_middle, int num_out);

void *nnMalloc(unsigned int size, int *err);
double nnGetRandWeight(int min, int max);
int nnAllocateMemory();
int nnReleaseMemory();

double nnGetErrorRate(double desiredOut[]);

int nnRunFeedForward(double valueIn[], double *valueOut);
int nnRunBackPropagation(double valueIn[], double desiredOut[]);

// Unreliable
double nnSelectiveFeedForward(double valueIn[], int numOut);
int nnSelectiveBackPropagation(double valueIn[], int numOut, double desiredOut);

// Deprecated
int nnSetUniqueFileID(int setID);

int nnReadWeightsFromFile(const char filename[]);
int nnSaveWeightsToFile(const char filename[]);

void nnBarf(char message[])
{
	printf("[OpenNN] %s\n", message);
}

int nnReturnNormal()
{
	return !OPEN_NET_ERROR;
}

int nnReturnError(char string[])
{
	nnBarf(string);

	return OPEN_NET_ERROR;
}

int nnPrintWeightInfo(int layer)
{
	if(layer & 1)
	{
		int i, j;
		
		for(i=0; i<NUM_IN; i++)
		{
			for(j=0; j<NUM_MID; j++)
				printf("%6.2f ", weight_L1[i][j]);
			
			printf("\n");
		}

		printf("\n");
	}

	if(layer & 2)
	{
		int j, k;
		
		for(j=0; j<NUM_MID; j++)
		{
			for(k=0; k<NUM_OUT; k++)
				printf("%6.2f ", weight_L2[j][k]);

			printf("\n");
		}

		printf("\n");
	}

	return nnReturnNormal();
}

double nnGetSingleWeightInfo(int layer, int percA, int percB)
{
	switch(layer)
	{
		case 1:
		{
			if(0<=percA && percA<NUM_IN && 0<=percB && percB<NUM_MID)
				return weight_L1[percA][percB];
		}
		case 2:
		{
			if(0<=percA && percA<NUM_MID && 0<=percB && percB<NUM_OUT)
				return weight_L2[percA][percB];
		}

		default:
		{
			nnBarf("[error] Null return: <nnGetSingleWeightInfo> layer not allocated\n");
			return 0;
		}
	}

	nnBarf("Layer not initialised");
	return 0;
}

double nnGetSingleNetResult(int numOut)
{
	if(0<=numOut && numOut<NUM_OUT)
	{
		return sigmoidOut_L2[numOut];
	}
	else
	{
		nnBarf("No perceptron at the location");
		return 0;
	}
}

void nnPrintTest()
{
	printf("\n>> Hello Brain?\n");

	printf("[size] %d %d %d\n", NUM_IN, NUM_MID, NUM_OUT);
}

int nnSetNetworkSize(int num_in, int num_middle, int num_out)
{
	if(num_in>0 && num_middle>0 && num_out>0)
	{
		NUM_IN = num_in;
		NUM_MID = num_middle;
		NUM_OUT = num_out;

		return nnReturnNormal();
	}
	else
	{
		NUM_IN = 0;
		NUM_MID = 0;
		NUM_OUT = 0;

		return nnReturnError("Failed to initialise network dimensions: out of range");
	}
}

double nnGetRandWeight(int min, int max)
{
	double temp;
	const int scale = 100;

	min *= scale;
	max *= scale;

	temp = rand() % (max-min);
	temp += min;
	temp /= scale;

	return temp;
}


void *nnMalloc(unsigned int size, int *err) // error-checked
{
	void *ptrArr = malloc(size);

	if(ptrArr == NULL)
	{
		nnBarf("Error occurred while allocating memory");
		*err |= 1;
	}

	return ptrArr;
}

int nnAllocateMemory()
{
	int i, j, k;
	int err = 0;

	// set weights on layer 1
	weight_L1 = (double**)malloc(NUM_IN * sizeof(double*));

	for(i=0; i<NUM_IN; i++)
	{
		weight_L1[i] = (double*)malloc(NUM_MID * sizeof(double));
	
		for(j=0; j<NUM_MID; j++)
		{
			weight_L1[i][j] = nnGetRandWeight(-1, 1);
		}
	}

	// set weights on layer 2
	weight_L2 = (double**)malloc(NUM_MID * sizeof(double*));

	for(j=0; j<NUM_MID; j++)
	{
		weight_L2[j] = (double*)malloc(NUM_OUT * sizeof(double));

		for(k=0; k<NUM_OUT; k++)
		{
			weight_L2[j][k] = nnGetRandWeight(-1, 1);
		}
	}

	productSum_L1 = (double*)nnMalloc(NUM_MID * sizeof(double), &err);
	productSum_L2 = (double*)nnMalloc(NUM_OUT * sizeof(double), &err);

	sigmoidOut_L1 = (double*)nnMalloc(NUM_MID * sizeof(double), &err);
	sigmoidOut_L2 = (double*)nnMalloc(NUM_OUT * sizeof(double), &err);

	delta2 = (double*)nnMalloc(NUM_MID * sizeof(double), &err);
	delta1 = (double*)nnMalloc(NUM_OUT * sizeof(double), &err);

	return nnReturnNormal();
}

int nnReleaseMemory()
{
	for(int i=0; i<NUM_IN; i++)
		free(weight_L1[i]);

	for(int j=0; j<NUM_MID; j++)
		free(weight_L2[j]);

	free(weight_L1);
	free(weight_L2);

	free(productSum_L1);
	free(productSum_L2);

	free(sigmoidOut_L1);
	free(sigmoidOut_L2);

	free(delta2);
	free(delta1);

	return nnReturnNormal();
}

double nnSigmoid(double x)
{
	return 1.0/(1 + exp(-x));
}

double nnGetErrorRate(double desiredOut[])
{
	int k;
	double totalErr = 0, temp;

	for(k=0; k<NUM_OUT; k++)
	{
		temp = desiredOut[k] - nnGetSingleNetResult(k);
		totalErr += temp * temp;
	}

	totalErr /= NUM_OUT;

	return totalErr;
}

int nnRunFeedForward(double valueIn[], double *valueOut)
{
	int i, j, k;

	// layer 1
	for(j=0; j<NUM_MID; j++)
	{
		productSum_L1[j] = 0.0;

		for(i=0; i<NUM_IN; i++)
			productSum_L1[j] += valueIn[i] * weight_L1[i][j];

		sigmoidOut_L1[j] = nnSigmoid(productSum_L1[j]);
	}

	// layer 2
	for(k=0; k<NUM_OUT; k++)
	{
		productSum_L2[k] = 0.0;

		for(j=0; j<NUM_MID; j++)
			productSum_L2[k] += sigmoidOut_L1[j] * weight_L2[j][k];

		sigmoidOut_L2[k] = nnSigmoid(productSum_L2[k]);
	}

	// copy the result
	for(k=0; k<NUM_OUT; k++)
	{
		valueOut[k] = sigmoidOut_L2[k];
	}

	return nnReturnNormal();
}

double nnSelectiveFeedForward(double valueIn[], int numOut)
{
	int i, j, k;

	// layer 1
	for(j=0; j<NUM_MID; j++)
	{
		productSum_L1[j] = 0.0;

		for(i=0; i<NUM_IN; i++)
			productSum_L1[j] += valueIn[i] * weight_L1[i][j];

		sigmoidOut_L1[j] = nnSigmoid(productSum_L1[j]);
	}

	// layer 2
	productSum_L2[numOut] = 0.0;

	for(j=0; j<NUM_MID; j++)
		productSum_L2[numOut] += sigmoidOut_L1[j] * weight_L2[j][numOut];

	sigmoidOut_L2[numOut] = nnSigmoid(productSum_L2[numOut]);

	return sigmoidOut_L2[numOut];
}

int nnRunBackPropagation(double valueIn[], double desiredOut[])
{
	int i, j, k;

	for(k=0; k<NUM_OUT; k++)
	{
		delta1[k] = (desiredOut[k] - sigmoidOut_L2[k]) * (sigmoidOut_L2[k] * (1-sigmoidOut_L2[k]));

		for(j=0; j<NUM_MID; j++)
		{
			weight_L2[j][k] += delta1[k] * sigmoidOut_L1[j];
			delta2[j] = delta1[k] * weight_L2[j][k] * (sigmoidOut_L1[j] * (1-sigmoidOut_L1[j]));

			for(i=0; i<NUM_IN; i++)
			{
				weight_L1[i][j] += delta2[j] * valueIn[i];
			}
		}
	}

	return nnReturnNormal();
}

int nnSelectiveBackPropagation(double valueIn[], int numOut, double desiredOut)
{
	int i, j, k = numOut;

	delta1[k] = (desiredOut - sigmoidOut_L2[k]) * (sigmoidOut_L2[k] * (1-sigmoidOut_L2[k]));

	for(j=0; j<NUM_MID; j++)
	{
		weight_L2[j][k] += delta1[k] * sigmoidOut_L1[j];
		delta2[j] = delta1[k] * weight_L2[j][k] * (sigmoidOut_L1[j] * (1-sigmoidOut_L1[j]));

		for(i=0; i<NUM_IN; i++)
		{
			weight_L1[i][j] += delta2[j] * valueIn[i];
		}
	}

	return nnReturnNormal();
}


void nnPlantMutationSeeds(double threshold, double amount)
{
	int i, j, k;
	double sign;

	threshold = fabs(threshold);
	amount = fabs(amount);

	for(i=0; i<NUM_IN; i++)
	{
		for(j=0; j<NUM_MID; j++)
		{
			sign = (weight_L1[i][j] > 0)? 1 : -1;

			if(sign * weight_L1[i][j] >= threshold)
				weight_L1[i][j] -= sign * nnGetRandWeight(0, amount);
		}
	}

	for(j=0; j<NUM_MID; j++)
	{
		for(k=0; k<NUM_OUT; k++)
		{
			sign = (weight_L2[j][k] > 0)? 1 : -1;

			if(sign * weight_L2[j][k] >= threshold)
				weight_L2[j][k] -= sign * nnGetRandWeight(0, amount);
		}
	}
}

int nnSetUniqueFileID(int setID)
{
	FILE_ID = setID;

	return nnReturnNormal();
}

int nnReadWeightsFromFile(const char filename[])
{
	if(FILE_ID == -1)
		return nnReturnError("");

	if(filename[0] == 0)
		return nnReturnError("File name not initialised");

	// check the directory
	int i, j, k;
	int fileCode, numIn, numMid, numOut;

	char location[64];
	sprintf(location, "%s/%s.txt", directory, filename);

	FILE *fpRead = fopen(location, "rt");

	if(fpRead == NULL)
		return nnReturnError("Failed to open an external file");

	// check the network dimension
	fscanf(fpRead, "%d %d %d\n", &numIn, &numMid, &numOut);

	if(numIn != NUM_IN || numMid != NUM_MID || numOut != NUM_OUT)
		return nnReturnError("Failed to load weights: different net size");

	// read from file
	for(i=0; i<NUM_IN; i++)
	{
		for(j=0; j<NUM_MID; j++)
		{
			fscanf(fpRead, "%lf\n", &weight_L1[i][j]);
		}
	}

	for(j=0; j<NUM_MID; j++)
	{
		for(k=0; k<NUM_OUT; k++)
		{
			fscanf(fpRead, "%lf\n", &weight_L2[j][k]);
		}
	}

	fclose(fpRead);

	return nnReturnNormal();
}

int nnSaveWeightsToFile(const char filename[])
{
	if(filename[0] == 0)
		return nnReturnError("File name not initialised");

	// open external file
	int i, j, k;

	char location[64];
	sprintf(location, "%s/%s.txt", directory, filename);

	FILE *fpWrite = fopen(location, "wt");

	if(fpWrite == NULL)
		return nnReturnError("Failed to generate custom file");

	// save net dimensions
	fprintf(fpWrite, "%d %d %d\n", NUM_IN, NUM_MID, NUM_OUT);

	// layer 1
	for(i=0; i<NUM_IN; i++)
	{
		for(j=0; j<NUM_MID; j++)
		{
			fprintf(fpWrite, "%16.8f\n", weight_L1[i][j]);
		}
	}

	// layer 2
	for(j=0; j<NUM_MID; j++)
	{
		for(k=0; k<NUM_OUT; k++)
		{
			fprintf(fpWrite, "%16.8f\n", weight_L2[j][k]);
		}
	}

	fclose(fpWrite);

	return nnReturnNormal();
}
