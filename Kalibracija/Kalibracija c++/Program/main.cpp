
/* 

	main.cpp
	
	Description: An example driver program for distortion.cpp. It simply creates
	an LUT from calibration parameters or undistorts and image given an LUT.

	Author: Chris Broaddus (cbroaddu@utk.ed)

	Notes: I can't guarantee any of this! If you do decide to use this and have 
	problems I will try to help. 

  */

#include "distortion.h"
#include "highgui.h"
#include <string.h>
#include <stdio.h>

// -l CalibResults.txt lut.txt
// -u lut.txt iqeye_wide1.bmp iqeye_correct1.bmp

void PrintOptions()
{
	printf("undistort -l params lut\n");
	printf("\tGenerates a lookup table (LUT) from params and save in lut\n");
	printf("undistort -u lut inimage outimage\n");
	printf("\tUndistorts inimage using the lut and saves it in outimage\n");
}

int main(int argc, char** argv)
{
	CvMat *LUT = 0;
	CvMat *A = 0;
	CvMat *Kappa = 0;
	CvMat *Rho = 0;
	int width, height;

	if(argc == 1)
	{
		PrintOptions();

		return 0;
	}

	if(!strcmp(argv[1], "-l"))
	{
		IRIS_LoadCalibParams(argv[2], &A, &Kappa, &Rho, width, height);
		LUT = IRIS_MakeLUT(A, Kappa, Rho, width, height);
		IRIS_SaveLUT(LUT, argv[3], width, height);
	}
	if(!strcmp(argv[1], "-u"))
	{
		IplImage* undistImg;
		IplImage* distImg;

		LUT = IRIS_LoadLUT(argv[2]);
		distImg = cvLoadImage(argv[3], 0);
		undistImg = IRIS_Undistort(distImg, LUT);
		cvSaveImage(argv[4], undistImg);

        cvNamedWindow( "Image view", 1 );
        cvShowImage( "Image view", undistImg );
        cvWaitKey(0); // very important
        cvDestroyWindow( "Image view" );
        cvReleaseImage( &undistImg );

		cvReleaseImage(&distImg);
		cvReleaseImage(&undistImg);
	}

	return 0;
}
