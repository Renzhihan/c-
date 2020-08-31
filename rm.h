#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include<vector>
#include<array>
#include<algorithm>

using namespace cv;
using namespace std;

namespace R {
	enum Colors
	{
		BLUE = 0,
		GREEN = 1,
		RED = 2
	};

	struct ArmorParam
	{	
		float light_min_area =10 ;
		float light_max_angle = 45;
		float light_contour_min_solidity = 0.2;
		float light_max_angle_diff = 10.0;
		float light_max_height_diff_ratio = 0.4;
		float light_max_y_diff_ratio = 1.0;
		float light_max_x_diff_ratio = 3.0;
		int enemy_color = BLUE;
		Size srcImageSize = Size(600, 800);

	} _param;

	class LightDescriptor
	{
	public:
		LightDescriptor() {}

		LightDescriptor(const RotatedRect& light)
		{
			width = light.size.width;
			length = light.size.height;
			center = light.center;

			if (light.angle > 135.0)
				angle = light.angle - 180.0;
			else
				angle = light.angle;
		}
		RotatedRect rec() const
		{
			return RotatedRect(center, Size2f(width, length), angle);
		}

	public:
		float width;
		float length;
		Point2f center;
		float angle;
	};


	class ArmorDescriptor
	{
	public:

		ArmorDescriptor() {}

		ArmorDescriptor(const LightDescriptor& lLight,
			const LightDescriptor& rLight)
		{
			lightPairs[0] = lLight.rec();
			lightPairs[1] = rLight.rec();

			cv::Size exLSize(int(lightPairs[0].size.width), int(lightPairs[0].size.height * 2));
			cv::Size exRSize(int(lightPairs[1].size.width), int(lightPairs[1].size.height * 2));
			cv::RotatedRect exLLight(lightPairs[0].center, exLSize, lightPairs[0].angle);
			cv::RotatedRect exRLight(lightPairs[1].center, exRSize, lightPairs[1].angle);

			cv::Point2f pts_l[4];
			exLLight.points(pts_l);
			cv::Point2f upper_l = pts_l[2];
			cv::Point2f lower_l = pts_l[3];

			cv::Point2f pts_r[4];
			exRLight.points(pts_r);
			cv::Point2f upper_r = pts_r[1];
			cv::Point2f lower_r = pts_r[0];

			vertex.resize(4);
			vertex[0] = upper_l;
			vertex[1] = upper_r;
			vertex[2] = lower_r;
			vertex[3] = lower_l;

			center.x = (upper_l.x + lower_r.x) / 2;
			center.y = (upper_l.y + lower_r.y) / 2;

		}

	public:
		std::array<cv::RotatedRect, 2> lightPairs;	   
		std::array<int, 2> lightsFlags;				  
		std::vector<cv::Point2f> vertex;			 	
		cv::Point2f center;						    

		float sizeScore;		//S1 = e^(size)
		float distScore;		//S2 = e^(-offset)
		float finalScore;		//sum of all the scores


	};

	class ArmorDetector 
	{

	public:
		Mat srcImage;
		int enemycolor;
		vector<ArmorDescriptor> armors;
		vector<ArmorDescriptor> True_armors;
		ArmorDescriptor target; 
		int _flag;

		enum DetectorFlag
		{
			ARMOR_NO = 0, 		
			ARMOR_FOUND = 1   
		};

		ArmorDetector() {};

		void setColor(int enemy_color)
		{
			enemycolor = enemy_color;
		}

		int detect(const Mat& srcImage)
		{
			armors.clear();
			vector<LightDescriptor> lightInfos;

			vector<Mat> channels;
			Mat image;
			
			split(srcImage, channels);
			
			Mat bImage = channels[0];
			Mat gImage = channels[1];
			Mat rImage = channels[2];
			if (enemycolor == BLUE)	subtract(channels[0], channels[2], image);
			else subtract(channels[2], channels[0], image);
			threshold(image, image, 80, 255, cv::THRESH_BINARY);
			imshow("ss", image);
			vector<vector<Point>> lightContours;
			vector<float>height;

			cv::findContours(image, lightContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			for (const auto& contour : lightContours)
			{
				float lightContourArea = contourArea(contour);

				if (contour.size() <= 5)	continue;

				RotatedRect lightRec = fitEllipse(contour);	
		
				if (lightContourArea < _param.light_min_area) continue;
				if (lightContourArea / lightRec.size.area() < _param.light_contour_min_solidity) continue;
				if (lightRec.angle < 170 && lightRec.angle>10) continue;
				
				height.push_back(lightRec.size.height);
				int maxheight = *max_element(height.begin(), height.end());
				if (lightRec.size.height * 1.2 < maxheight) continue;
				
				lightInfos.emplace_back(lightRec);
			}		

			if (lightInfos.empty())	return ARMOR_NO;

			sort(lightInfos.begin(), lightInfos.end(), [](const LightDescriptor& ld1, const LightDescriptor& ld2)
				{
					return ld1.center.x < ld2.center.x;
				});
			
			for (int i = 0; i < lightInfos.size(); i++)
			{
				for (int j = i + 1; j < lightInfos.size(); j++)
				{
					const LightDescriptor& leftLight = lightInfos[i];
					const LightDescriptor& rightLight = lightInfos[j];

					float angleDifferent = abs(leftLight.angle - rightLight.angle);
					if (angleDifferent> _param.light_max_angle_diff) continue; //	(5,10)

					float Lenth_ratio = abs(leftLight.length - rightLight.length) / (leftLight.length> rightLight.length?leftLight.length:rightLight.length);  
					if (Lenth_ratio > _param.light_max_height_diff_ratio)	continue;

					float meanLen = (leftLight.length + rightLight.length) ;//2

					float yDifferent = abs(leftLight.center.y - rightLight.center.y);
					float yDifferent_ratio = yDifferent / meanLen;
					if (yDifferent_ratio+yDifferent_ratio > _param.light_max_y_diff_ratio) continue;

					float xDifferent = abs(leftLight.center.x - rightLight.center.x);
					float xDifferent_ratio = xDifferent / meanLen;
					if (xDifferent_ratio+xDifferent_ratio > _param.light_max_x_diff_ratio) continue;
					
					ArmorDescriptor armor(leftLight, rightLight);

					armor.lightsFlags[0] = i;
					armor.lightsFlags[1] = j;
					armors.emplace_back(armor);
					break;
				}
			}

			if (armors.empty())	return ARMOR_NO;			

			for (int i = 0; i < armors.size(); i++)
			{
				bool true_or_false = 1;
				if (armors.size() > 2)
				{
					for (int j = i + 1; j < armors.size(); j++)
					{
						if (armors[i].lightsFlags[0] == armors[j].lightsFlags[0] ||
							armors[i].lightsFlags[0] == armors[j].lightsFlags[1] ||
							armors[i].lightsFlags[1] == armors[j].lightsFlags[0] ||
							armors[i].lightsFlags[1] == armors[j].lightsFlags[1])
						{
							true_or_false = 0;
							if (abs(armors[i].vertex[0].x - armors[i].vertex[1].x) > abs(armors[j].vertex[0].x - armors[j].vertex[1].x)) True_armors.emplace_back(armors[i]);
							else  True_armors.emplace_back(armors[j]);
							i++;
							break;
						}
					}
				}
				if (true_or_false)	True_armors.emplace_back(armors[i]);
			}

			if (True_armors.empty()) return ARMOR_NO;

			target = True_armors[0];

			return ARMOR_FOUND;
		}

		void draw(Mat src)
		{						
				for (int i = 0; i < 4; i++)
				{
					line(src, target.vertex[i], target.vertex[(i + 1) % 4], Scalar(200, 255, 0), 1, LINE_AA);
				}			
		}

	};
}
