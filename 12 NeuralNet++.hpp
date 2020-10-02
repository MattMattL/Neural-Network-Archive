//	Class-based 2-layer neural networks, written in C++
//
//	A class that has a neural network "package" for programs that
//	requires multiple neural networks running in parallel, e.g. image
//	processing with RGB values.
//
//	(sadly, this version uses Malloc not New)
//

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iomanip>

#define NPP_RAND_SEED time(NULL)
#define NPP_TERMINATED -1

using namespace std;

class NNet
{
private:
	int NET_IN, NET_MID, NET_OUT;

	double *productSum_L1, *productSum_L2;
	double *sigmoidOut_L1;

	double **weight_L1, **weight_L2;
	double *delta1, *delta2;

	string name;
	string directory;

	double* nppOneDimMalloc(int size) // error-checked memory allocation
	{
		double *ptrTemp = (double*)malloc(size * sizeof(double));

		if(ptrTemp == NULL)
			nppTerminate("nppOneDimMalloc", "Error occurred while allocating memory");

		return ptrTemp;
	}

	double** nppTwoDimMalloc(int row, int column) // error-checked memory allocation
	{
		double **ptrTemp = (double**)malloc(row * sizeof(double*));

		if(ptrTemp == NULL)
			nppTerminate("nppTwoDimMalloc", "Error occurred while allocating memory (1)");

		for(int j=0; j<row; j++)
			ptrTemp[j] = nppOneDimMalloc(column);

		return ptrTemp;
	}

	double nppRNG(int lower, int upper)
	{
		double temp;
		const double scale = 10000;

		upper *= scale;
		lower *= scale;

		temp = rand() % (upper - lower);
		temp += lower;
		temp /= scale;

		return temp;
	}

// Print Error Message

	void nppBarf(string message)
	{
		cout << "[system] " << message << endl;
	}

	// print error and terminate
	void nppTerminate(string location, string error)
	{
		cout << "[error] " << location << "::" << error << endl;
		exit(1);
	}

public:

	double *input, *output, *desired;

// Preinitialisation

	// set a unique network name
	void nppSetNNetName(string nameIn)
	{
		name = nameIn;
	}

	// set directory path
	void nppSetNNetDirectory(string locationIn)
	{
		directory = locationIn + '/';
	}

	// or use current directory as default
	void nppSetNNetDirectory()
	{
		directory = "./";
	}

	// set network size
	void nppSetNNetSize(int num_in, int num_mid, int num_out)
	{
		if(num_in < 1 || num_mid < 1 || num_out < 1)
			nppTerminate("nppSetNNetSize", "Network size cannot be initialised");

		NET_IN  = num_in;
		NET_MID = num_mid;
		NET_OUT = num_out;
	}

	// allocate memory. all networks commands to be called after this function
	void nppAllocateMemory()
	{
		int i, j, k;

		input = nppOneDimMalloc(NET_IN);
		output = nppOneDimMalloc(NET_OUT);
		desired = nppOneDimMalloc(NET_OUT);

		productSum_L1 = nppOneDimMalloc(NET_MID);
		productSum_L2 = nppOneDimMalloc(NET_OUT);

		sigmoidOut_L1 = nppOneDimMalloc(NET_MID);

		weight_L1 = nppTwoDimMalloc(NET_IN, NET_MID);
		weight_L2 = nppTwoDimMalloc(NET_MID, NET_OUT);

		delta2 = nppOneDimMalloc(NET_MID);
		delta1 = nppOneDimMalloc(NET_OUT);
	}

	// initialise weights with random values
	void nppSetRandWeights(int min, int max)
	{
		srand(NPP_RAND_SEED);

		for(int i=0; i<NET_IN; i++)
			for(int j=0; j<NET_MID; j++)
				weight_L1[i][j] = nppRNG(min, max);

		for(int j=0; j<NET_MID; j++)
			for(int k=0; k<NET_OUT; k++)
				weight_L2[j][k] = nppRNG(min, max);
	}

	void nppSetRandWeights()
	{
		srand(NPP_RAND_SEED);

		for(int i=0; i<NET_IN; i++)
			for(int j=0; j<NET_MID; j++)
				weight_L1[i][j] = nppRNG(-1, 1);

		for(int j=0; j<NET_MID; j++)
			for(int k=0; k<NET_OUT; k++)
				weight_L2[j][k] = nppRNG(-1, 1);
	}

	void nppReleaseMemory()
	{
		free(input);
		free(output);
		free(desired);

		free(productSum_L1);
		free(productSum_L2);

		free(sigmoidOut_L1);

		for(int j=0; j<NET_IN; j++)
			free(weight_L1[j]);

		for(int k=0; k<NET_MID; k++)
			free(weight_L2[k]);

		free(weight_L1);
		free(weight_L2);

		free(delta2);
		free(delta1);
	}

	double nppSigmoid(double x)
	{
		return 1.0 / (1 + exp(-x));
	}

	void nppRunFeedForward()
	{
		int i, j, k;

		// layer 1
		for(j=0; j<NET_MID; j++)
		{
			productSum_L1[j] = 0;

			for(i=0; i<NET_IN; i++)
				productSum_L1[j] += input[i] * weight_L1[i][j];

			sigmoidOut_L1[j] = nppSigmoid(productSum_L1[j]);
		}

		// layer 2
		for(k=0; k<NET_OUT; k++)
		{
			productSum_L2[k] = 0;

			for(j=0; j<NET_MID; j++)
				productSum_L2[k] += sigmoidOut_L1[j] * weight_L2[j][k];

			output[k] = nppSigmoid(productSum_L2[k]);
		}
	}

	void nppRunBackProp()
	{
		int i, j, k;

		for(k=0; k<NET_OUT; k++)
		{
			delta1[k] = (desired[k] - output[k]) * (output[k] * (1-output[k]));

			for(j=0; j<NET_MID; j++)
			{
				weight_L2[j][k] += delta1[k] * sigmoidOut_L1[j];
				delta2[j] = delta1[k] * weight_L2[j][k] * (sigmoidOut_L1[j] * (1-sigmoidOut_L1[j]));

				for(i=0; i<NET_IN; i++)
				{
					weight_L1[i][j] += delta2[j] * input[i];
				}
			}
		}
	}

	double nppGetWeightValue(int layerIn, int row, int column)
	{
		switch(layerIn)
		{
			case 1:
			{
				if(0<=row && row<NET_IN && 0<=column && column<NET_MID)
					return weight_L1[row][column];
			}
			case 2:
			{
				if(0<=row && row<NET_MID && 0<=column && column<NET_OUT)
					return weight_L2[row][column];
			}

			default:
			{
				nppTerminate("nppGetWeightInfo", "Network layer requested not available");
			}
		}

		nppTerminate("nppGetWeightInfo", "Perceptron requested not available");

		return NPP_TERMINATED;
	}

	//legacy
	double nppGetNNetResult(int arr)
	{
		if(0<=arr && arr<NET_OUT)
			return output[arr];
		else
			nppTerminate("nppGetNNetResult", "Perceptron not initialised");

		return NPP_TERMINATED;
	}

	void nppPrintNNetResult()
	{
		int max = (NET_IN > NET_OUT)? NET_IN : NET_OUT;

		for(int n=0; n<max; n++)
		{
			if(n < NET_IN)
				printf("%5.0f", input[n]);
			else
				printf("     ");

			printf("  -->  ");

			if(n < NET_OUT)
				printf("%4.2f  (%1.0f)\n", output[n], desired[n]);
			else
				printf("\n");
		}

		printf("\n");
	}

	void nppPrintWeights(int layer_1, int layer_2)
	{
		if(layer_1)
			{
			for(int i=0; i<NET_IN; i++)
			{
				for(int j=0; j<NET_MID; j++)
					printf("%6.2f ", weight_L1[i][j]);

				cout << endl;
			}

			cout << endl;
		}

		if(layer_2)
		{
			for(int j=0; j<NET_MID; j++)
			{
				for(int k=0; k<NET_OUT; k++)
					printf("%6.2f ", weight_L2[j][k]);

				cout << endl;
			}

			cout << endl;
		}
	}


// File IO

	void nppLoadWeightsFromFile()
	{
		if(name.empty())
			nppTerminate("nppLoadWeightsFromFile", "Unique name not initialised");

		int num_in, num_mid, num_out;
		string location;

		ifstream fsRead;

		location = directory + name + ".txt";
		fsRead.open(location);

		if(!fsRead)
			nppTerminate("nppLoadWeightsFromFile", "Unable to open file");

		fsRead >> num_in >> num_mid >> num_out;

		cout << "[NNet] network size: ";
		cout << num_in << ", " << num_mid << ", " << num_out << endl;

		if(num_in != NET_IN || num_mid != NET_MID || num_out != NET_OUT)
		{
			nppBarf("Incompatible network dimensions, re-initialising weights...");
			nppSetRandWeights();
			return;
		}

		for(int i=0; i<NET_IN; i++)
			for(int j=0; j<NET_MID; j++)
				fsRead >> weight_L1[i][j];

		for(int j=0; j<NET_MID; j++)
			for(int k=0; k<NET_OUT; k++)
				fsRead >> weight_L2[j][k];
	}

	void nppSaveWeightsToFile()
	{
		if(name.empty())
			nppTerminate("nppSaveWeightsToFile", "Network name not initialised");

		string location;
		ofstream fsWrite;

		location = directory + name + ".txt";
		fsWrite.open(location, ios::trunc);

		if(!fsWrite)
			nppTerminate("nppSaveWeightsToFile", "Unable to open the file");

		fsWrite << NET_IN << " " << NET_MID << " " << NET_OUT << endl;

		for(int i=0; i<NET_IN; i++)
			for(int j=0; j<NET_MID; j++)
				fsWrite << weight_L1[i][j] << endl;

		for(int j=0; j<NET_MID; j++)
			for(int k=0; k<NET_OUT; k++)
				fsWrite << weight_L2[j][k] << endl;

		fsWrite.close();
	}

};




