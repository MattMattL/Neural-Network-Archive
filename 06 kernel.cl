//	A kernel to support the OpenCL version of shallow neural nets.,
//	cf. multiCoreNNet.c
//

__kernel
void kernelTester(
	__global float *memOut,
	__global float *memRow,
	__global float *memCol,
	const int height,
	const int width)
{
    int id = get_global_id(0);

    memOut[id] += memRow[id/width] * memCol[id%width];
}


__kernel
void kernelBackPropDelta(
	__global float *memD,
	__global int   *memO,
	__global float *memZ,
	const int train,
	const int out)
{	
	int id = get_global_id(0);

	memD[id] = (memO[out*train+id] - memZ[id]) * (memZ[id]*(1-memZ[id]));
}


__kernel
void kernelBackPropLayer2(
	__global float *memW2,
	__global float *memD,
	__global float *memY,
	const int out)
{
    int id = get_global_id(0);
    
    memW2[id] += memD[id%out] * memY[id/out];
}


__kernel
void kernelBackPropLayer1(
	__global float *memW1,
	__global float *memW2,
	__global float *memD,
	__global float *memY,
	__global int *memX,
	const int train,
	const int in,
	const int mid,
	const int out)
{
	int id = get_global_id(0);

	memW1[id/out] += memD[id%out] * memW2[id%(mid*out)] * (memY[id%(mid*out) / out]*(1-memY[id%(mid*out) / out])) * memX[in*train + (id/out)/mid];
}
