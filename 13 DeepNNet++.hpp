//	Class-based 3 layer neural networks in C++
//
//	A deep neural network version of the class-based 2 layer neural
//	networks.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <math.h>
#include <time.h>

#define NUM_SAMPLE 1

using namespace std;

class DeepNNet
{
private:

	string location;

	// network size
	int NUM_IN, NUM_MID_L1, NUM_MID_L2, NUM_OUT;

	// weights
	double weightL1[NUM_IN][NUM_MID_L1];
	double weightL2[NUM_MID_L1][NUM_MID_L2];
	double weightL3[NUM_MID_L2][NUM_OUT];

	// sum of weighted values
	double pL1[NUM_MID_L1];
	double pL2[NUM_MID_L2];
	double pL3[NUM_OUT];

	// sigmoid out
	double yL1[NUM_MID_L1];
	double yL2[NUM_MID_L2];

	// delta
	double delta1[NUM_OUT];
	double delta2[NUM_MID_L2];
	double delta3[NUM_MID_L1];

	double sigmoid(double);
	double setSingleWeight();

public:

	// neural io
	double netIn[NUM_IN];
	double netOut[NUM_OUT];

	// desired out
	int desired[NUM_OUT];

	void initialiseWeights(int);
	void setFileLocation(string);
	void setNetworkSize(int, int, int, int);

	void printWeightInfo();
	double printNNetInfo();

	void runFeedForward();
	void runBackPropagation();

	int readWeightsFromFile();
	int writeWeightsToFile();

};

double DeepNNet::sigmoid(double x)
{
	return 1.0/(1 + exp(-x));
}

double DeepNNet::setSingleWeight()
{
	double weight;

	weight = rand() % 200 + 1;
	weight /= 100;
	weight -= 1;

	return weight;
}

void DeepNNet::initialiseWeights(int shouldReset)
{
	if(shouldReset)
	{
		int i, j;

		for(i=0; i<NUM_IN; i++)
			for(j=0; j<NUM_MID_L1; j++)
				weightL1[i][j] = setSingleWeight();

		for(i=0; i<NUM_MID_L1; i++)
			for(j=0; j<NUM_MID_L2; j++)
				weightL2[i][j] = setSingleWeight();

		for(i=0; i<NUM_MID_L2; i++)
			for(j=0; j<NUM_OUT; j++)
				weightL3[i][j] = setSingleWeight();
	}
	else
	{
		readWeightsFromFile();
	}
}

void DeepNNet::setFileLocation(string locationIn)
{
	location = locationIn;
}

void DeepNNet::setNetworkSize(int in, int mid1, int mid2, int out)
{
	NUM_IN = in;
	NUM_MID_L1 = mid1;
	NUM_MID_L2 = mid2;
	NUM_OUT = out;
}

void DeepNNet::printWeightInfo()
{
	int i, j;

	for(i=0; i<NUM_IN; i++)
	{
		for(j=0; j<NUM_MID_L1; j++)
			printf("%5.2lf ", weightL1[i][j]);
		printf("\n");
	}
	printf("\n");

	for(i=0; i<NUM_MID_L1; i++)
	{
		for(j=0; j<NUM_MID_L2; j++)
			printf("%5.2lf ", weightL2[i][j]);
		printf("\n");
	}
	printf("\n");

	for(i=0; i<NUM_MID_L2; i++)
	{
		for(j=0; j<NUM_OUT; j++)
			printf("%5.2lf ", weightL3[i][j]);
		printf("\n");
	}
	printf("\n");

	printf("\n");
}

double DeepNNet::printNNetInfo()
{
	int size_net;

	double diff;
	double sum = 0;

	printf("[Net]");
	runFeedForward();

	size_net = (NUM_OUT > NUM_IN)? NUM_OUT : NUM_IN;

	for(int iterate=0; iterate<size_net; iterate++)
	{
		if(iterate < NUM_IN)
			printf("\t%5lf", netIn[iterate]);
		else
			printf("\t     ");

		printf("  -->  ");

		if(iterate < NUM_OUT)
			printf("%4.2lf (%d)\n", netOut[iterate], desired[iterate]);
		else
			printf("\n");

		if(iterate < NUM_OUT)
		{
			diff = desired[iterate] - netOut[iterate];
			sum += diff * diff;
		}
	}

	printf("\n");

	return sum;
}

void DeepNNet::runFeedForward()
{
	int i, j, k, l;

	for(j=0; j<NUM_MID_L1; j++)
	{
		pL1[j] = 0;

		for(i=0; i<NUM_IN; i++)
			pL1[j] += netIn[i] * weightL1[i][j];

		yL1[j] = sigmoid(pL1[j]);
	}

	for(k=0; k<NUM_MID_L2; k++)
	{
		pL2[k] = 0;

		for(j=0; j<NUM_MID_L1; j++)
			pL2[k] += yL1[j] * weightL2[j][k];

		yL2[k] = sigmoid(pL2[k]);
	}

	for(l=0; l<NUM_OUT; l++)
	{
		pL3[l] = 0;

		for(int k=0; k<NUM_MID_L2; k++)
			pL3[l] += yL2[k] * weightL3[k][l];

		netOut[l] = sigmoid(pL3[l]);
	}
}

void DeepNNet::runBackPropagation()
{
	int i, j, k, l;

	for(l=0; l<NUM_OUT; l++)
	{
		delta1[l] = (desired[l] - netOut[l]) * (netOut[l] * (1-netOut[l]));

		for(k=0; k<NUM_MID_L2; k++)
		{
			weightL3[k][l] += delta1[l] * yL2[k];
			delta2[k] = delta1[l] * weightL3[k][l] * (yL2[k] * (1-yL2[k]));

			for(j=0; j<NUM_MID_L1; j++)
			{
				weightL2[j][k] += delta2[k] * yL1[j];
				delta3[j] = delta2[k] * weightL2[j][k] * (yL1[j] * (1-yL1[j]));

				for(i=0; i<NUM_IN; i++)
				{
					weightL1[i][j] += delta3[j] * netIn[i];
				}
			}
		}
	}
}

int DeepNNet::readWeightsFromFile()
{
	// open file
	ifstream fsRead;

	fsRead.open(location);

	if(!fsRead)
	{
		cout << "[error] no such external file found" << endl;
		exit(1);
	}
	
	// do not read if network size is different
	int in, mid1, mid2, out, set;
 	
 	fsRead >> in >> mid1 >> mid2 >> out >> set;

	if(in!=NUM_IN || mid1!=NUM_MID_L1 || mid2!=NUM_MID_L2 || out!=NUM_OUT || set!=NUM_SAMPLE)
	{
		cout << "[error] net size does not match. initialising new weights..." << endl;
		return 0;
	}

	// load from file
	int i, j, k;

	for(i=0; i<NUM_IN; i++)
		for(j=0; j<NUM_MID_L1; j++)
			fsRead >> weightL1[i][j];

	for(i=0; i<NUM_MID_L1; i++)
		for(j=0; j<NUM_MID_L2; j++)
			fsRead >> weightL2[i][j];

	for(i=0; i<NUM_MID_L2; i++)
		for(j=0; j<NUM_OUT; j++)
			fsRead >> weightL3[i][j];

	fsRead.close();

	return 1;
}

int DeepNNet::writeWeightsToFile()
{
	ofstream fsWrite;

	fsWrite.open(location, ios::trunc);

	if(!fsWrite)
	{
		cout << "[error] failed to generate a new external file" << endl;
		return 0;
	}

	fsWrite << NUM_IN 		<< " ";
	fsWrite << NUM_MID_L1 	<< " ";
	fsWrite << NUM_MID_L2 	<< " ";
	fsWrite << NUM_OUT 		<< " ";
	fsWrite << NUM_SAMPLE 	<< endl;

	int i, j;

	for(i=0; i<NUM_IN; i++)
		for(j=0; j<NUM_MID_L1; j++)
			fsWrite << weightL1[i][j] << endl;

	for(i=0; i<NUM_MID_L1; i++)
		for(j=0; j<NUM_MID_L2; j++)
			fsWrite << weightL2[i][j] << endl;

	for(i=0; i<NUM_MID_L2; i++)
		for(j=0; j<NUM_OUT; j++)
			fsWrite << weightL3[i][j] << endl;

	fsWrite.close();

	return 1;
}
