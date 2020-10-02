//	The very first version of neural net I tested
//
//	This one-layer neural networks was created for checking the mathematics
//	and equations for the feed forward and back propagation algorithm. The
//	process is hard coded.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double weight[2];

double input[2] = {0, 1};
double output[2] = {0, 1};

void setWeight()
{
	for(int i=0; i<2; i++)
	{
		weight[i] = rand() % 201; // 0 - 200
		weight[i] /= 100; // 0 - 2
		weight[i] -= 1; // -1 ~ 1
	}
}

double sigmoid(double x)
{
	return 1.0 / (1 + exp(-x));
}

int main()
{
	double x, y, z, d;
	double partialW1, partialW2;

	int info = 0;
	int max = 100000;

	srand(time(NULL));

	setWeight();

	for(int i=0; i<max; i++)
	{
		// run the net
		x = input[i%2];
		d = output[i%2];

		y = sigmoid(x * weight[0]);
		z = sigmoid(y * weight[1]);

		if(info || i<5 || (max-i)<=5)
		{
			printf("%5.2f  %5.2f  --> ", x, weight[0]);
			printf("%5.2f, %5.2f  -->  ", y, weight[1]);
			printf("%.4f (%4.2f) ", z, d);
		}

		//back propagation
		weight[1] += (d-z) * y * (z*(1-z));
		weight[0] += (d-z) * (z*(1-z)) * weight[1] * (y*(1-y)) * x;

		if(info || i < 5 || (max - i) <= 5)
		{
			printf("| %7.4f  %7.4f\n", weight[0], weight[1]);

			if(i==4)
				printf("\n");
		}
	}

	return 0;
}
