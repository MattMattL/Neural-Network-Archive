//	A header file for image processing, written in C++
//
//	This file is (partially) written in OpenCV for image processing for
//	personal projects. Includes functions for sobel edge detection, kernel
//	convolution and max/min/avg pooling.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "kernel++.hpp"

#define PIXEL uint8_t

using namespace cv;

Mat imgReturnImgBase(const int rows, const int cols)
{
	return Mat::zeros(rows, cols, CV_8UC1);
}

Mat imgReturnImgBase(const int rows, const int cols, const int initialVal)
{
	int i, j;

	Mat tempImage = Mat::zeros(rows, cols, CV_8UC1);

	for(i=0; i<rows; i++)
		for(j=0; j<cols; j++)
			tempImage.at<PIXEL>(i, j) = initialVal;

	return tempImage;
}

Mat imgRunConvolutions(Mat imageIn, Kernel kernel)
{
	int i, j, k_i, k_j;
	double sum;

	Mat imageOut = imgReturnImgBase(imageIn.rows - kernel.rows + 1, imageIn.cols - kernel.cols + 1);

	for(i=0; i<imageOut.rows; i++)
	{
		for(j=0; j<imageOut.cols; j++)
		{
			sum = 0;

			for(k_i=0; k_i<kernel.rows; k_i++)
			{
				for(k_j=0; k_j<kernel.rows; k_j++)
				{
					sum += imageIn.at<PIXEL>(i + k_i, j + k_j) * kernel.at[k_i][k_j];
				}
			}

			imageOut.at<PIXEL>(i, j) = sum;
		}
	}

	return imageOut;
}

Mat imgSobelEdgeDetection(Mat imageIn)
{
	const int SIZE_KERNEL = 3;

	int i, j, k_i, k_j;
	double sumHor, sumVer;

	int kernelHor[SIZE_KERNEL][SIZE_KERNEL];
	int kernelVer[SIZE_KERNEL][SIZE_KERNEL] = 
	{
		{1,0,-1},
		{2,0,-2},
		{1,0,-1}
	};

	Mat imageOut = imgReturnImgBase(imageIn.rows - SIZE_KERNEL + 1, imageIn.cols - SIZE_KERNEL + 1);

	// flip the vertical kernel to create the horrizontal kernel
	for(i=0; i<SIZE_KERNEL; i++)
		for(j=0; j<SIZE_KERNEL; j++)
			kernelHor[i][j] = kernelVer[j][i];

	// do kernel convolution
	for(i=0; i<imageOut.rows; i++)
	{
		for(j=0; j<imageOut.cols; j++)
		{
			sumHor = 0;
			sumVer = 0;

			for(k_i=0; k_i<SIZE_KERNEL; k_i++)
			{
				for(k_j=0; k_j<SIZE_KERNEL; k_j++)
				{
					sumHor += imageIn.at<PIXEL>(i+k_i, j+k_j) * kernelHor[k_i][k_j];
					sumVer += imageIn.at<PIXEL>(i+k_i, j+k_j) * kernelVer[k_i][k_j];
				}
			}

			imageOut.at<PIXEL>(i, j) = sqrt(sumVer*sumVer + sumHor*sumHor);
		}
	}

	return imageOut;
}

Mat imgRunAvgPooling(Mat imageIn, int ratioRow, int ratioCol)
{
	int i, j, pool_i, pool_j;
	double sum;

	Mat imageOut = imgReturnImgBase(imageIn.rows/ratioRow, imageIn.cols/ratioCol);

	for(i=0; i<imageOut.rows; i++)
	{
		for(j=0; j<imageOut.cols; j++)
		{
			sum = 0;

			for(pool_i=0; pool_i<ratioRow; pool_i++)
				for(pool_j=0; pool_j<ratioCol; pool_j++)
					sum += imageIn.at<PIXEL>(i*ratioRow + pool_i, j*ratioCol + pool_j);

			sum /= ratioRow * ratioCol;

			imageOut.at<PIXEL>(i, j) = sum;
		}
	}

	return imageOut;
}

Mat imgRunMaxPooling(Mat imageIn, int ratioRow, int ratioCol)
{
	int i, j, pool_i, pool_j;
	double max;

	Mat imageOut = imgReturnImgBase(imageIn.rows/ratioRow, imageIn.cols/ratioCol);

	for(i=0; i<imageOut.rows; i++)
	{
		for(j=0; j<imageOut.cols; j++)
		{
			max = imageIn.at<PIXEL>(i * ratioRow, j * ratioCol);

			for(pool_i=0; pool_i<ratioRow; pool_i++)
			{
				for(pool_j=0; pool_j<ratioCol; pool_j++)
				{
					if(!pool_i && !pool_j)
						continue;

					if(max < imageIn.at<PIXEL>(i*ratioRow + pool_i, j*ratioCol + pool_j))
						max = imageIn.at<PIXEL>(i*ratioRow + pool_i, j*ratioCol + pool_j);
				}
			}

			imageOut.at<PIXEL>(i, j) = max;
		}
	}

	return imageOut;
}
