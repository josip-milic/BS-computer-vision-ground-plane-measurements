
/* 

	distortion.h

	Author: Chris Broaddus (cbroaddu@utk.ed)

	Notes: I can't guarantee any of this! If you do decide to use this and have 
	problems I will try to help. 

  */

#include <stdio.h>
#include <math.h>
#include <cv.h>

IplImage* IRIS_Undistort(IplImage*, CvMat*);
CvMat* IRIS_MakeLUT(CvMat*, CvMat*, CvMat*, int , int);
void IRIS_SaveLUT(CvMat*, char*, int, int);
CvMat* IRIS_LoadLUT(char* filename);
void IRIS_LoadCalibParams(char*, CvMat**, CvMat**, CvMat**, int &, int &);