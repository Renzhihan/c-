#include"robomaster.h"
#include <ml.h>
#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include <cxcore.h>

using namespace cv;
using namespace std;
using namespace ml;
using namespace R;

int main()
{
	Mat src;
	double  timeSum = 0, timeAverage = 0;
	int  frameNum = 0;
	double  t, tc;
	VideoCapture cap("C:/Users/75741/Desktop/Armor/videos/car1 79fps kk.mp4");
	while(1)
	{
		t = getTickCount();
		{
			//src = imread("C:/Users/75741/Desktop/Armor/images/near/1 (3).bmp");
			cap >> src;
			if (src.empty())
			{
				printf("load image error");
				return 0;
			}
			ArmorDetector detector;
			detector.setColor(RED);	//choose armor color
			
			int flag = detector.detect(src);
			if (flag) 
			{
				detector.draw(src);
			}
			cv::imshow("装甲板识别", src);
			tc = (getTickCount() - t) / getTickFrequency();
			printf("ALL  the  time  consume  %.5f\n", tc);      
			printf("########    %f      fps    ######\n\n", double((double)1.0 / (double)tc));
			waitKey(1);			
		}	
	}
	return 0;

}
