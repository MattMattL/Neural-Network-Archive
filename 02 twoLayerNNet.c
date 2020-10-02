//	First test version of two-layer neural networks
//
//	The input and output vectors are declared as arrays, although the size
//	was practically fixed due to the equations being hard coded.
//
//<-------------------- 50 --------------------->| -------- 72 ------->|

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int input[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
int out[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};

double weight[2][4];

void printWeight();

void setWeight()
{

	printf("\n# initial set\n");
	
	for(int i=0; i<2; i++)
	{
		for(int j=0; j<4; j++)
		{
			weight[i][j] = rand()%2001; // 0 - 20
			weight[i][j] /= 100; // 0.00 - 2.00

			weight[i][j] -= 10; // -1 - 1
		}
	}

	printWeight();
}

double sigmoid(double x)
{
	return 1.0 / (1 + exp(-x));
}

void printWeight()
{
	for(int i=0; i<2; i++)
	{
		printf("[weights] ");

		for(int j=0; j<4; j++)
			printf("%8.4f ", weight[i][j]);

		printf("\n");
	}

	printf("\n");
}

int main()
{
	double x1, x2;
	double y1, y2;
	double z1, z2;
	double d1, d2;

	int max = 1000000;
	int info = 0, set = 4;

	srand(time(NULL));

	setWeight();

	for(int i=0; i<max/set; i++)
	{
		for(int j=0; j<set; j++)
		{
			// run feed forward
			x1 = input[j][0];
			x2 = input[j][1];

			y1 = sigmoid(x1 * weight[0][0] + x2 * weight[0][1]);
			y2 = sigmoid(x1 * weight[0][2] + x2 * weight[0][3]);

			z1 = sigmoid(y1 * weight[1][0] + y2 * weight[1][1]);
			z2 = sigmoid(y1 * weight[1][2] + y2 * weight[1][3]);
			
			if(info || (max/set-i)<=1)
			{
				if(!j)
					printf("[out]  ");
				else
					printf("       ");

				printf("%7.4f %7.4f  --> %7.4f %7.4f\n", x1, x2, z1, z2);

				if(j == set - 1)
					printWeight();
			}

			// back propagation
			weight[1][0] += (out[j][0]-z1) * (z1*(1-z1)) * y1;
			weight[1][1] += (out[j][0]-z1) * (z1*(1-z1)) * y2;
			weight[1][2] += (out[j][0]-z1) * (z2*(1-z2)) * y1;
			weight[1][3] += (out[j][1]-z2) * (z2*(1-z2)) * y2;

			weight[0][0] += (out[j][0]-z1) * z1*(1-z1) * weight[1][0] * y1*(1-y1) * x1;
			weight[0][0] += (out[j][0]-z2) * z2*(1-z2) * weight[1][2] * y1*(1-y1) * x1;
			weight[0][1] += (out[j][1]-z1) * z1*(1-z1) * weight[1][0] * y1*(1-y1) * x2;
			weight[0][1] += (out[j][1]-z2) * z2*(1-z2) * weight[1][2] * y1*(1-y1) * x2;
			weight[0][2] += (out[j][0]-z1) * z1*(1-z1) * weight[1][1] * y1*(1-y1) * x1;
			weight[0][2] += (out[j][0]-z2) * z2*(1-z2) * weight[1][3] * y1*(1-y1) * x1;
			weight[0][3] += (out[j][1]-z1) * z1*(1-z1) * weight[1][1] * y1*(1-y1) * x2;
			weight[0][3] += (out[j][1]-z2) * z2*(1-z2) * weight[1][3] * y1*(1-y1) * x2;
		}
	}

	return 0;
}
