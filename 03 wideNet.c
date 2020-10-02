//	Second version of shallow network
//
//	Arrays are used to simplify the feed forward and back propagation
//	process and to support different network size, width-wise. And
//	using Delta was also tested for optimisation.
//
//	File IO were added to save the weights so the network can be trained
//	repeatedly.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int x[16][4]  = {{0,0,0,1}, {0,0,1,0}, {0,1,0,0}, {1,0,0,0}};
int Pf[16][4] = {{0,0,0,1}, {0,0,1,0}, {0,1,0,0}, {1,0,0,0}};

double p[4];
double y[4];
double z[4];

double weight[2][4][4];
double delta[4];

int loadPreset();
int saveWeightVals();

int setInitialWeight()
{
	double temp;

	for(int layer=0; layer<2; layer++)
	{
		for(int i=0; i<4; i++)
		{
			for(int j=0; j<4; j++)
			{
				temp = rand() % 200 + 1; // 1 - 200;
				temp /= 100; // 0.01 - 2.00
				temp -= 1; // -1 - 1

				weight[layer][i][j] = temp;
			}
		}
	}

	return 0;
}

double sigmoid(double x)
{
	return 1.0/(1 + exp(-x));
}

int printWeightInfo()
{
	printf("\nWeights:\n");

	for(int i=0; i<4; i++)
	{
		for(int j=0; j<4; j++)
			printf("    %10.4f  %10.4f\n", weight[0][i][j], weight[1][i][j]);

		printf("\n");
	}

	return 0;
}

int printNetOut()
{
	for(int set=0; set<4; set++)
	{
		// Simulate Net
		for(int i=0; i<4; i++)
		{
			p[i] = 0;

			for(int j=0; j<4; j++)
				p[i] += x[set][j] * weight[0][j][i];

			y[i] = sigmoid(p[i]);
		}

		for(int i=0; i<4; i++)
		{
			p[i] = 0;

			for(int j=0; j<4; j++)
				p[i] += y[j] * weight[1][j][i];

			z[i] = sigmoid(p[i]);
		}

		// Print Info
		printf("[Out]");

		for(int i=0; i<4; i++)
			printf("\t%d  -->  %7.2f (%d)\n", x[set][i], z[i], Pf[set][i]);

		printf("\n");
	}

	return 0;
}

int main()
{
	int err;
	double temp;

	double time_s=0;
	time_s = clock();

	srand(time(NULL));

	err = setInitialWeight();
	//err = loadPreset();

	// Initialisation
	printf("# Before Training\n");
	printWeightInfo();
	printNetOut();

	// Settings
	int max = 1000000;
	int stage = 4;

	for(int round=max/stage; round>0; round--)
	{
		for(int set=stage-1; set>=0; set--)
		{
			// Feed Forward
			for(int i=0; i<4; i++)
			{
				p[i] = 0;

				for(int j=0; j<4; j++)
					p[i] += x[set][j] * weight[0][j][i];
			}

			for(int i=0; i<4; i++)
				y[i] = sigmoid(p[i]);

			for(int i=0; i<4; i++)
			{
				p[i] = 0;

				for(int j=0; j<4; j++)
					p[i] += y[j] * weight[1][j][i];
			}

			for(int i=0; i<4; i++)
				z[i] = sigmoid(p[i]);

			// Back Prop'
			for(int k=0; k<4; k++)
			{
				delta[k] = (Pf[set][k]-z[k]) * (z[k]*(1-z[k]));

				for(int j=0; j<4; j++)
				{
					weight[1][j][k] += delta[k] * y[j];

					for(int i=0; i<4; i++)
					{
						weight[0][i][j] += delta[k] * weight[1][j][k] * (y[j]*(1-y[j])) * x[set][i];
					}
				}
			}
		}
	}

	printf("# Result\n");
	printWeightInfo();
	printNetOut();

	saveWeightVals();

	time_s = clock() - time_s;
	time_s /= CLOCKS_PER_SEC;
	printf("execution time : %.3f s\n\n", time_s);

	return 0;
}

const char *location = "./nNet/weight.txt";

int loadPreset()
{
	FILE *fpRead = fopen(location, "rt");

	if(fpRead == NULL)
		return 1;

	for(int layer=0; layer<2; layer++)
		for(int i=0; i<4; i++)
			for(int j=0; j<4; j++)
				fscanf(fpRead, "%lf\n", &weight[layer][i][j]);

	fclose(fpRead);

	return 0;
}

int saveWeightVals()
{
	FILE *fpWrite = fopen(location, "wt");

	if(fpWrite == NULL)
		return 1;

	for(int layer=0; layer<2; layer++)
		for(int i=0; i<4; i++)
			for(int j=0; j<4; j++)
				fprintf(fpWrite, "%12f\n", weight[layer][i][j]);

	fclose(fpWrite);

	return 0;
}
