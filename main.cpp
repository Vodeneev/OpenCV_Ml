#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include"opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
using namespace cv;
using namespace ml;
using namespace std;


int main() 
{
	int clusterCount = 3;
	cv::Ptr<cv::ml::TrainData> data_set = cv::ml::TrainData::loadFromCSV("C:/Users/1/Documents/Date/beverage_r.csv", 1, 0, 1, "ord[0]cat[1-8]", ';', '?');
	cv::Mat Matd = data_set->getTrainSamples();
	cv::Mat labels;
	cv::Mat centers(clusterCount, 1, Matd.type());
	kmeans(Matd, clusterCount, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10000, 0.0001), 50, cv::KMEANS_PP_CENTERS, centers);
	cout << centers;
	return 0;
}