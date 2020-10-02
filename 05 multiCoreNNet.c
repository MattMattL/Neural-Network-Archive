//	OpenCL version of the shallow neural networks
//
//	This version uses OpenCL for multi-core processing. Average execution
//	time was decreased (no specific percentage calculated, tested with a
//	4-thread CPU) when the net width was over 1024, compared to other single-
//	core versions.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#define NUM_IN 128
#define NUM_MID 128
#define NUM_OUT 128
#define NUM_SET 4

#define MAX_SOURCE_SIZE 0x100000

const int in = NUM_IN;
const int mid = NUM_MID;
const int out = NUM_OUT;

float weightL1[NUM_IN][NUM_MID];
float weightL2[NUM_MID][NUM_OUT];

int x[NUM_SET][NUM_IN]
	 = {{0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,1,1,1,1,0,0,0,0,1,1,1,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,1,0,1,1,0,0,0}, 
		{0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,1,0,0,1,0,0,0,1,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,0,1,0,0,0,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0}, 
		{1,1,1,0,0,1,1,0,1,0,0,0,1,1,1,1,0,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,0,0,0,0,1,1,0,1,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0}, 
		{1,0,0,0,0,1,1,1,1,1,1,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,0,1,0,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,1,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0}};

int pOut[NUM_SET][NUM_OUT]
	 = {{0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,1,1,1,1,0,0,0,0,1,1,1,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,1,0,1,1,0,0,0}, 
		{0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,1,0,0,1,0,0,0,1,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,0,1,0,0,0,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0}, 
		{1,1,1,0,0,1,1,0,1,0,0,0,1,1,1,1,0,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,0,0,0,0,1,1,0,1,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0}, 
		{1,0,0,0,0,1,1,1,1,1,1,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,0,1,0,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,1,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0}};

float pL1[NUM_MID];
float pL2[NUM_OUT];
float y[NUM_MID];
float z[NUM_OUT];

float delta[NUM_OUT];

float *ptrWL1 = weightL1[0];
float *ptrWL2 = weightL2[0];

int   *ptrX = x[0];
float *ptrY = y;
float *ptrZ = z;
int   *ptrOut = pOut[0];

float *ptrDelta = delta;

int printWeight();
int printNNOut();
int loadWeights();
int saveWeights();

float sigmoid(float x)
{
	return 1.0 / (1 + exp(-x));
}

int printWeight(int flag1, int flag2)
{
	int i, j, k;

	if(flag1)
	{
		printf("[weights 1]");
		for(i=0; i<NUM_IN; i++)
		{
			printf("\t");

			for(j=0; j<NUM_MID; j++)
				printf("%8.4f ", weightL1[i][j]);

			printf("\n");
		}
	}

	if(flag2)
	{
		printf("[weights 2]");
		for(j=0; j<NUM_MID; j++)
		{
			printf("\t");

			for(k=0; k<NUM_OUT; k++)
				printf("%8.4f ", weightL2[j][k]);

			printf("\n");
		
		}
	}

	printf("\n");

	return 0;
}

int printNNOut()
{
	int train, i, j, k;

	for(train=0; train<NUM_SET; train++)
	{
		for(j=0; j<NUM_MID; j++)
		{
			pL1[j] = 0;

			for(i=0; i<NUM_IN; i++)
				pL1[j] += x[train][i] * weightL1[i][j];

			y[j] = sigmoid(pL1[j]);
		}

		for(k=0; k<NUM_OUT; k++)
		{
			pL2[k] = 0;

			for(j=0; j<NUM_MID; j++)
				pL2[k] += y[j] * weightL2[j][k];

			z[k] = sigmoid(pL2[k]);
		}

		// Print Result //
		printf("[Net]");

		int iteration = (NUM_IN <= NUM_OUT)? NUM_OUT : NUM_IN;

		for(i=0; i<iteration; i++)
		{
			if(i<NUM_IN)
				printf("\t%5d", x[train][i]);
			else
				printf("\t     ");

			printf("  -->  ");

			if(i<NUM_OUT)
				printf("%4.2f (%d)\n", z[i], pOut[train][i]);
			else
				printf("\n");
		}

		printf("\n");
	}

	return 0;
}

int initialiseWeight()
{
	float temp;
	int i, j, k;

	for(i=0; i<NUM_IN; i++)
	{
		for(j=0; j<NUM_MID; j++)
		{
			temp = rand()%200 + 1; //1-200
			temp /= 100; //0.01-2.00
			temp -= 1; //-1 - 1

			weightL1[i][j] = temp;
		}
	}

	for(j=0; j<NUM_MID; j++)
	{
		for(k=0; k<NUM_OUT; k++)
		{
			temp = rand()%200 + 1; //1-200
			temp /= 100; //0.01-2.00
			temp -= 1; //-1 - 1

			weightL2[j][k] = temp;
		}
	}

	return 0;
}

int main()
{
	/* Initialising Kernel and Memory */

	cl_program program[4];
	cl_kernel kernel[4];

	cl_command_queue command_queue;
	cl_context context;
	cl_device_id cpu = NULL, device = NULL;
	cl_int clerr = 0;

	size_t local = 0;
	size_t global = 0;

	cl_mem memX, memY, memZ, memOut;
	cl_mem memWL1, memWL2, memDelta;

	clerr = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device, NULL);

	// To use a GPU:
	//
	// clerr = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	// if(clerr != CL_SUCCESS)
	// {
	// 	device = cpu;
	// }
	
	context = clCreateContext(0, 1, &device, NULL, NULL, &clerr);
	command_queue = clCreateCommandQueue(context, device, 0, &clerr);

	const char fileName[] = "./kernel.cl";
	FILE *fp = fopen(fileName, "r");

	if(fp == NULL)
		printf("\n\n[error] no readable kernel found\n\n\n");

	char *source_str;
	size_t source_size;

	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);

	fclose(fp);

	program[0] = clCreateProgramWithSource(context, 1, (const char**) &source_str, (const size_t*) &source_size, &clerr);
	clerr = clBuildProgram(program[0], 0, NULL, NULL, NULL, NULL);
	kernel[0] = clCreateKernel(program[0], "kernelBackPropDelta", &clerr);

	program[1] = clCreateProgramWithSource(context, 1, (const char**) &source_str, (const size_t*) &source_size, &clerr);
	clerr = clBuildProgram(program[1], 0, NULL, NULL, NULL, NULL);
	kernel[1] = clCreateKernel(program[1], "kernelBackPropLayer2", &clerr);

	program[2] = clCreateProgramWithSource(context, 1, (const char**) &source_str, (const size_t*) &source_size, &clerr);
	clerr = clBuildProgram(program[2], 0, NULL, NULL, NULL, NULL);
	kernel[2] = clCreateKernel(program[2], "kernelBackPropLayer1", &clerr);

	/* Neural Networks */

	int round, train;
	int i, j, k;
	int err = 3;

	srand(1010);

	initialiseWeight();
	//err = loadWeights();

	if(err==3)
		printf("[system] Weights initialised\n\n");
	else if(err==1)
		printf("[error] File location not found\n\n");
	else if(err==2)
		printf("[error] Net size not matched\n\n");
	else if(err==0)
		printf("[system] Loading weight info\n\n");

	//setting
	int iteration = 100;
	int info = 0;

	if(info)
	{
		printWeight(1, 1);
		printNNOut();
	}

	double runtime = 0.0;
	runtime = clock();

	for(round=0; round<(iteration/NUM_SET); round++)
	{
		if(!info)
			printf("%4d/%d\n", round+1, iteration/NUM_SET);

		for(train=0; train<NUM_SET; train++)
		{
			if(!info)
				printf("	%4d/%d\n", train+1, NUM_SET);
			
			/* Feed Forward */

			for(j=0; j<NUM_MID; j++)
			{
				pL1[j] = 0.0;

				for(i=0; i<NUM_IN; i++)
					pL1[j] += x[train][i] * weightL1[i][j];

				y[j] = sigmoid(pL1[j]);
			}

			for(k=0; k<NUM_OUT; k++)
			{
				pL2[k] = 0.0;

				for(j=0; j<NUM_MID; j++)
					pL2[k] += y[j] * weightL2[j][k];

				z[k] = sigmoid(pL2[k]);
			}

			/* Back Prop: Delta */

			memDelta = clCreateBuffer(context, CL_MEM_READ_WRITE,			(NUM_OUT) * sizeof(float)	, NULL, NULL);
			memOut   = clCreateBuffer(context, CL_MEM_READ_WRITE,	(NUM_SET*NUM_OUT) * sizeof(int)		, NULL, NULL);
			memZ     = clCreateBuffer(context, CL_MEM_READ_WRITE,			(NUM_OUT) * sizeof(float)	, NULL, NULL);
			
			clerr  = clEnqueueWriteBuffer(command_queue, memDelta	, CL_TRUE, 0,			(NUM_OUT) * sizeof(float),	ptrDelta, 0, NULL, NULL);					
			clerr |= clEnqueueWriteBuffer(command_queue, memOut		, CL_TRUE, 0,	(NUM_SET*NUM_OUT) * sizeof(int),	ptrOut,	  0, NULL, NULL);
			clerr |= clEnqueueWriteBuffer(command_queue, memZ		, CL_TRUE, 0,			(NUM_OUT) * sizeof(float),	ptrZ,	  0, NULL, NULL);
			clFinish(command_queue);

			clerr  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &memDelta);
			clerr |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &memOut);
			clerr |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &memZ);
			clerr |= clSetKernelArg(kernel[0], 3, sizeof(cl_int), &train);
			clerr |= clSetKernelArg(kernel[0], 4, sizeof(cl_int), &out);

			clerr = clGetKernelWorkGroupInfo(kernel[0], device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
			
			global = NUM_OUT;
			//local = global / 2;

			clerr = clEnqueueNDRangeKernel(command_queue, kernel[0], 1, NULL, &global, &local, 0, NULL, NULL);
			clFinish(command_queue);

			clerr = clEnqueueReadBuffer(command_queue, memDelta, CL_TRUE, 0, (NUM_OUT) * sizeof(float), ptrDelta, 0, NULL, NULL);
			clFinish(command_queue);

			clReleaseMemObject(memOut);
			clReleaseMemObject(memZ);


			/* Back Prop: Layer2 */

			memWL2   = clCreateBuffer(context, CL_MEM_READ_WRITE, (NUM_MID*NUM_OUT) * sizeof(float), NULL, NULL);
			memY     = clCreateBuffer(context, CL_MEM_READ_WRITE,  		  (NUM_MID) * sizeof(float), NULL, NULL);
			//memD reused
			
			clerr  = clEnqueueWriteBuffer(command_queue, memWL2,	CL_TRUE, 0, (NUM_MID*NUM_OUT) * sizeof(float), ptrWL2,   0, NULL, NULL);
			clerr |= clEnqueueWriteBuffer(command_queue, memDelta,			CL_TRUE, 0, (NUM_OUT) * sizeof(float), ptrDelta, 0, NULL, NULL);
			clerr |= clEnqueueWriteBuffer(command_queue, memY,				CL_TRUE, 0, (NUM_MID) * sizeof(float), ptrY,	 0, NULL, NULL);
			clFinish(command_queue);

			clerr  = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &memWL2);
			clerr |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &memDelta);
			clerr |= clSetKernelArg(kernel[1], 2, sizeof(cl_mem), &memY);
			clerr |= clSetKernelArg(kernel[1], 3, sizeof(cl_int), &out);

			clerr = clGetKernelWorkGroupInfo(kernel[1], device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
			
			global = NUM_MID * NUM_OUT;
			// local = global / 2;

			clerr = clEnqueueNDRangeKernel(command_queue, kernel[1], 1, NULL, &global, &local, 0, NULL, NULL);
			clFinish(command_queue);

			clerr = clEnqueueReadBuffer(command_queue, memWL2, CL_TRUE, 0, (NUM_MID*NUM_OUT) * sizeof(float), ptrWL2, 0, NULL, NULL);
			clFinish(command_queue);

			clReleaseMemObject(memDelta);
			clReleaseMemObject(memY);

			/* Back Prop - Adjusting Weights on Layer 1 */

			memWL1   = clCreateBuffer(context, CL_MEM_READ_WRITE,	 (NUM_IN*NUM_MID) * sizeof(float), NULL, NULL);
			memDelta = clCreateBuffer(context, CL_MEM_READ_WRITE,			(NUM_OUT) * sizeof(float), NULL, NULL);
			memY 	 = clCreateBuffer(context, CL_MEM_READ_WRITE,			(NUM_MID) * sizeof(float), NULL, NULL);
			memX 	 = clCreateBuffer(context, CL_MEM_READ_WRITE,	 (NUM_IN*NUM_SET) * sizeof(int)	 , NULL, NULL);
			// memWL2 reused

			clerr  = clEnqueueWriteBuffer(command_queue, memWL1		, CL_TRUE, 0, 	 (NUM_IN*NUM_MID) * sizeof(float), ptrWL1,	 0, NULL, NULL);
			clerr |= clEnqueueWriteBuffer(command_queue, memWL2		, CL_TRUE, 0, 	(NUM_MID*NUM_OUT) * sizeof(float), ptrWL2,	 0, NULL, NULL);
			clerr |= clEnqueueWriteBuffer(command_queue, memDelta	, CL_TRUE, 0, 			(NUM_OUT) * sizeof(float), ptrDelta, 0, NULL, NULL);
			clerr |= clEnqueueWriteBuffer(command_queue, memY		, CL_TRUE, 0, 			(NUM_MID) * sizeof(float), ptrY,	 0, NULL, NULL);
			clerr |= clEnqueueWriteBuffer(command_queue, memX		, CL_TRUE, 0, 	 (NUM_IN*NUM_SET) * sizeof(int),   ptrX,	 0, NULL, NULL);
			clFinish(command_queue);

			clerr  = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &memWL1);
			clerr |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &memWL2);
			clerr |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), &memDelta);
			clerr |= clSetKernelArg(kernel[2], 3, sizeof(cl_mem), &memY);
			clerr |= clSetKernelArg(kernel[2], 4, sizeof(cl_mem), &memX);
			clerr |= clSetKernelArg(kernel[2], 5, sizeof(cl_int), &train);
			clerr |= clSetKernelArg(kernel[2], 6, sizeof(cl_int), &in);
			clerr |= clSetKernelArg(kernel[2], 7, sizeof(cl_int), &mid);
			clerr |= clSetKernelArg(kernel[2], 8, sizeof(cl_int), &out);

			clerr = clGetKernelWorkGroupInfo(kernel[2], device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
			
			global = NUM_IN * NUM_MID * NUM_OUT;
			//local = global / 2;

			clerr = clEnqueueNDRangeKernel(command_queue, kernel[2], 1, NULL, &global, &local, 0, NULL, NULL);
			clFinish(command_queue);

			clerr = clEnqueueReadBuffer(command_queue, memWL1, CL_TRUE, 0, (NUM_IN*NUM_MID) * sizeof(float), ptrWL1, 0, NULL, NULL);
			clFinish(command_queue);

			clReleaseMemObject(memWL1);
			clReleaseMemObject(memWL2);
			clReleaseMemObject(memDelta);
			clReleaseMemObject(memY);
			clReleaseMemObject(memX);
		}

		if(!info)
			printf("\n");
	}

	runtime = clock() - runtime;
	runtime /= CLOCKS_PER_SEC;

	if(info)
	{
		printWeight(1, 1);
		printNNOut();
	}
	printNNOut();
	
	//saveWeights();

	printf("execution time : %.3f s\n\n", runtime);

	clReleaseProgram(program[0]);
	clReleaseProgram(program[1]);
	clReleaseProgram(program[2]);
	
	clReleaseKernel(kernel[0]);
	clReleaseKernel(kernel[1]);
	clReleaseKernel(kernel[2]);

	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	printf("global = %zu, local = %zu\n\n", global, local);

	printf("\a");

	return 0;
}


const char *location = "./openWeight.txt";

int loadWeights()
{
	int input, middle, output, set;
	int i, j, k;

	FILE *fpRead = fopen(location, "rt");

	if(fpRead == NULL)
		return 1;

	fscanf(fpRead, "%d %d %d %d\n", &input, &middle, &output, &set);

	if(input!=NUM_IN || middle!=NUM_MID || output!=NUM_OUT || set!=NUM_SET)
		return 2;

	for(i=0; i<NUM_IN; i++)
		for(j=0; j<NUM_MID; j++)
			fscanf(fpRead, "%f\n", &weightL1[i][j]);


	for(j=0; j<NUM_MID; j++)
		for(k=0; k<NUM_OUT; k++)
			fscanf(fpRead, "%f\n", &weightL2[j][k]);

	return 0;
}


int saveWeights()
{
	int i, j, k;

	FILE *fpWrite = fopen(location, "wt");

	if(fpWrite == NULL)
		return 1;

	fprintf(fpWrite, "%d %d %d %d\n", NUM_IN, NUM_MID, NUM_OUT, NUM_SET);

	for(i=0; i<NUM_IN; i++)
		for(j=0; j<NUM_MID; j++)
			fprintf(fpWrite, "%10.4f\n", weightL1[i][j]);

	for(j=0; j<NUM_MID; j++)
		for(k=0; k<NUM_OUT; k++)
			fprintf(fpWrite, "%10.4f\n", weightL2[j][k]);

	return 0;
}
