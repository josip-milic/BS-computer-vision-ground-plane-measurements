#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>
#include <stdio.h>>

using namespace std;
//int main(int argc, char** argv)
int main(int argc)
{
	int i = 0;
	char image[1000];
	cvNamedWindow("mainWin");
	CvCapture* capture = cvCreateFileCapture("I:/DCIM/101GOPRO/GOPR2191.MP4");
	//CvCapture* capture = cvCreateFileCapture(argv[1]);
	int broj;
	int br = 0;
	IplImage* frame, *output;
	struct tm *local;
	time_t t;
	printf("Press 's' to save image 'Esc' to exit\n");
	while (1)
	{
		frame = cvQueryFrame(capture);
		if (!frame) break;
		cvShowImage("mainWin", frame);
		char c = cvWaitKey();
		if (c == 's')
		{
				output = cvCreateImage(cvGetSize(frame), 8, 3);
				br++;

				sprintf(image, "slika%d.png",br);
				cvSaveImage(image, frame);
				i++;
				printf("Slika je spremljena!");
		}
		
		
		else if (c == 27) 
		{
			break;
		}
		else
		{
			continue;
		}
	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("mainWin");
	return 0;
}

