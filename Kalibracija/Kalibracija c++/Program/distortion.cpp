
/* 

	distortion.cpp
	
	Description: Routines to make a lookup table (LUT), and save and write
	it to a file. There is also a lookup table to undistort and image given
	an LUT. 

	Author: Chris Broaddus (cbroaddu@utk.ed)

	Notes: I can't guarantee any of this! If you do decide to use this and have 
	problems I will try to help. 

  */

#include "distortion.h"

IplImage* IRIS_Undistort(IplImage* F, CvMat* LUT)
{
	// Verify bit depth is 8 bits before continuing 

	if(F->depth != IPL_DEPTH_8U)
		return NULL;

	double xp, yp;

	int width = F->width;
	int height = F->height;

	IplImage* G = cvCreateImage(cvSize(width, height), F->depth, F->nChannels);

	unsigned char* Fdata = (unsigned char*)F->imageData;
	unsigned char* Gdata = (unsigned char*)G->imageData;	
	float* LUTData = (float*)LUT->data.ptr;

	int p = 0;
	for(int x = 0; x < F->width; ++x)
	{
		for(int y = 0; y < F->height; ++y) 
		{
			// Get new coordinate from the LUT such
			// that (x,y) -> (xp, yp)

			xp = LUTData[p * 2];
			yp = LUTData[p * 2 + 1];

			// Get value at (x,p) from F and move to
			// G at (xp,yp) using nearest neighbor 
			// interpolation

			for(int k = 1; k <= F->nChannels; ++k)
			{
				Gdata[y * width * F->nChannels + x * F->nChannels + k] = 
					Fdata[cvRound(yp) * width * F->nChannels + cvRound(xp) * F->nChannels + k];
			}

			++p;
		}
	}

	return G;
}

CvMat* IRIS_MakeLUT(CvMat* A, CvMat* Kappa, CvMat* Rho, int width, int height)
{
	double xn, yn;
	double xp, yp;
	double xh, yh, zh;
	double xptmp, yptmp;
	double rn;

	int numRadialCoef = 0;
	int numDecenCoef = 0;

	if(Kappa != 0)
		numRadialCoef = Kappa->height;
	if(Rho != 0)
		numDecenCoef = Rho->height;

	// Allocate the LUT

	CvMat *LUT = cvCreateMat((height) * (width), 2, CV_32F);

	// Get inverse of calibration matrix for normalization

	CvMat *invA = cvCreateMat(3, 3, CV_32F);

	cvInvert(A, invA);

	// Extract intrinsic parameters

	double u = cvmGet(A, 0, 2);
	double v = cvmGet(A, 1, 2);
	double alpha = cvmGet(A, 0, 0);
	double beta = cvmGet(A, 1, 1);
	double s = cvmGet(A, 0, 1);

	cvmSet(A, 0, 2, u);
	cvmSet(A, 1, 2, v);

	int p = 0;
	for(int x = 0; x < width; ++x) 
	{
		for(int y = 0; y < height; ++y)
		{
			xp = x;
			yp = y;

			// Correct skew

			xp += yp * atan(s / beta);

			if(numRadialCoef > 0 || numDecenCoef > 0)
			{
				// Normalize coordinate using inverse of camera calibration matrix, i.e. mn = inv(A) * m

				xh = cvmGet(invA, 0, 0) * xp + cvmGet(invA, 0, 1) * yp + cvmGet(invA, 0, 2);
				yh = cvmGet(invA, 1, 0) * xp + cvmGet(invA, 1, 1) * yp + cvmGet(invA, 1, 2);
				zh = cvmGet(invA, 2, 0) * xp + cvmGet(invA, 2, 1) * yp + cvmGet(invA, 2, 2);

				xn = xh / zh;
				yn = yh / zh;

				// Compute normalized radius

				rn = sqrt(xn * xn + yn * yn);
			}

			// Radial distortion

			xptmp = xp;
			yptmp = yp;

			// Add linear term

			if(numRadialCoef > 0)
			{
				xp += (xptmp - u) * (cvmGet(Kappa, 1, 0) * rn);
				yp += (yptmp - v) * (cvmGet(Kappa, 1, 0) * rn);				
			}

			// Add nonlinear terms

			for(int i = 1; i < numRadialCoef; ++i)
			{
				xp += (xptmp - u) * (cvmGet(Kappa, i, 0) * pow(rn, 2 * i));
				yp += (yptmp - v) * (cvmGet(Kappa, i, 0) * pow(rn, 2 * i));
			}

			// Recompute rn using the radially adjusted coordinates for tangential distortion

			xh = cvmGet(invA, 0, 0) * xp + cvmGet(invA, 0, 1) * yp + cvmGet(invA, 0, 2);
			yh = cvmGet(invA, 1, 0) * xp + cvmGet(invA, 1, 1) * yp + cvmGet(invA, 1, 2);
			zh = cvmGet(invA, 2, 0) * xp + cvmGet(invA, 2, 1) * yp + cvmGet(invA, 2, 2);

			xn = xh / zh;
			yn = yh / zh;

			// Compute normalized radius

			rn = sqrt(xn * xn + yn * yn);

			// Decentering distortion

			if(numDecenCoef > 2)
			{
				xp += ((2 * (cvmGet(Rho, 0, 0)) * (xp - u) * (yp - v)) + cvmGet(Rho, 1, 0) * 
					((rn * rn) + 2 * ((xp - u) * (xp - u))));
				yp += ((2 * (cvmGet(Rho, 1, 0)) * (xp - u) * (yp - v)) + cvmGet(Rho, 0, 0) * 
					((rn * rn) + 2 * ((yp - v) * (yp - v))));

				xptmp = xp;
				yptmp = yp;
				for(int i = 2; i < numDecenCoef; ++i)
				{
					xp += ((2 * cvmGet(Rho, 0, 0) * (xptmp - u) * (yptmp - v)) + cvmGet(Rho, 1, 0) * 
						((rn * rn) + 2 * ((xptmp - u) * (xptmp - u)))) * 
						cvmGet(Rho, i, 0) * pow(rn, 2 * (i - 1));
					yp += ((2 * cvmGet(Rho, 1, 0) * (xptmp - u) * (yptmp - v)) + cvmGet(Rho, 0, 0) * 
						((rn * rn) + 2 * ((yptmp - v) * (yptmp - v)))) * 
						cvmGet(Rho, i, 0) * pow(rn, 2 * (i - 1));
				}
			}

			// Add to LUT

			cvmSet(LUT, p, 0, xp);
			cvmSet(LUT, p, 1, yp);

			++p;
		}
	}

	return LUT;
}

void IRIS_SaveLUT(CvMat* LUT, char* filename, int width, int height)
{
	FILE *fid = fopen(filename, "w");

	fprintf(fid, "%d %d\n", width, height);

	// Write LUT to file

	for(int i = 0; i < LUT->height; ++i)
		fprintf(fid, "%f %f\n", cvmGet(LUT, i, 0), cvmGet(LUT, i, 1));

	fclose(fid);
}

CvMat* IRIS_LoadLUT(char* filename)
{
	FILE *fid = fopen(filename, "r");

	int width, height;
	float xp, yp;

	fscanf(fid, "%d", &width);
	fscanf(fid, "%d", &height);

	// Allocat the LUT

	CvMat *LUT = cvCreateMat(height * width, 2, CV_32F);

	// Load LUT from file

	for(int p = 0; p < width * height; ++p)
	{
		fscanf(fid, "%f", &xp);
		fscanf(fid, "%f", &yp);

		cvmSet(LUT, p, 0, xp);
		cvmSet(LUT, p, 1, yp);
	}

	fclose(fid);

	return LUT;
}

void IRIS_LoadCalibParams(char* filename, CvMat** A, CvMat** Kappa, CvMat** Rho, int &width, int &height)
{
	FILE *fid = fopen(filename, "r");	

	// Load dimension

	fscanf(fid, "%d", &width);
	fscanf(fid, "%d", &height);
	
	// Load calibration matrix

	float alpha, beta, s, u, v;
	*A = cvCreateMat(3, 3, CV_32F);

	fscanf(fid, "%f", &alpha);
	fscanf(fid, "%f", &s);
	fscanf(fid, "%f", &u);
	fscanf(fid, "%f", &beta);
	fscanf(fid, "%f", &v);

	cvmSet(*A, 0, 0, alpha);
	cvmSet(*A, 0, 1, s);
	cvmSet(*A, 0, 2, u);
	cvmSet(*A, 1, 0, 0);
	cvmSet(*A, 1, 1, beta);
	cvmSet(*A, 1, 2, v);
	cvmSet(*A, 2, 0, 0);
	cvmSet(*A, 2, 1, 0);
	cvmSet(*A, 2, 2, 1);

	// Load radial distortion coeficients

	int numRadialCoef;
	float kappa;
	
	fscanf(fid, "%d", &numRadialCoef);

	if(numRadialCoef != 0)
	{
		*Kappa = cvCreateMat(numRadialCoef, 1, CV_32F);
		for(int i = 0; i < numRadialCoef; ++i)
		{
			fscanf(fid, "%f", &kappa);
			cvmSet(*Kappa, i, 0, kappa);
		}
	}

	// Load Decentering distortion coeficients

	int numDecenCoef;
	float rho;
	
	fscanf(fid, "%d", &numDecenCoef);

	if(numDecenCoef != 0)
	{
		*Rho = cvCreateMat(numDecenCoef, 1, CV_32F);
		for(int j = 0; j < numDecenCoef; ++j)
		{
			fscanf(fid, "%f", &rho);
			cvmSet(*Rho, j, 0, rho);
		}
	}
}