#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include"opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <locale.h>
using namespace cv;
using namespace ml;
using namespace std;


int main()
{
	setlocale(LC_ALL, "ru");
	string file_name = "C:/Users/1/Documents/Date/agaricus-lepiota.data";
	cv::Ptr<cv::ml::TrainData> data_set = cv::ml::TrainData::loadFromCSV(file_name, 0, 0, 1, "cat[0-22]", ',', '?');
	int samples = data_set->getNSamples();
	if (samples == 0) {
		cerr << "Ќе удалось прочитать файл: " << file_name << endl;
		exit(-1);
	}
	else {
		cout << "ѕрочитано " << samples << " примеров из файла " << file_name << endl;
	}

	data_set->setTrainTestSplitRatio(0.90, true);

	int n_train_samples = data_set->getNTrainSamples();
	int n_test_samples = data_set->getNTestSamples();
	cout << "Ќайдено " << n_train_samples << " обучающих примеров и "
		<< n_test_samples << " тестовых примеров" << endl;

	cv::Ptr<cv::ml::RTrees> dtree = cv::ml::RTrees::create();

	float _priors[] = { 1.0, 10.0 };

	cv::Mat priors(1, 2, CV_32F, _priors);
	dtree->setMaxDepth(5);
	dtree->setMinSampleCount(10);
	dtree->setRegressionAccuracy(0.01f);
	dtree->setUseSurrogates(false /* true */);
	dtree->setMaxCategories(3);
	dtree->setCVFolds(0);
	dtree->setUse1SERule(true);
	dtree->setTruncatePrunedTree(true);
	dtree->setPriors( priors );

	dtree->train(data_set);

	cv::Mat results;
	float train_performance = dtree->calcError(
		data_set,
		false, // использовать обучающие данные
		results // cv::noArray()
	);

	std::vector<cv::String> names;
	data_set->getNames(names);
	Mat flags = data_set->getVarSymbolFlags();
	float test_performance = dtree->calcError(
		data_set,
		true, // использовать тестовые данные
		results //cv::noArray()
	);

	cout << " ачество на обучающих данных: " << 100 - train_performance << endl;
	cout << " ачество на тестовых данных: " << 100 - test_performance << endl;
	std::vector<DTrees::Node> Nodes = dtree->getNodes();
	for (int i = 0; i < Nodes.size(); i++)
	{
		cout << Nodes[i].value << ' ';
	}
	return 0;
}














	/*int main()
	{
		int N = 50;
		int flag = 0;
		Mat X(N, 1, CV_32FC1);
		Mat ff(N, 1, CV_32FC1);
		float h = 3.14 / N;

		for (int i = 0; i < N; i++)
		{
			X.at<float>(i, 0) = i * h;
			ff.at<float>(i, 0) = sinf(i * h);
			cout << X.at<float>(i, 0) << " " << ff.at<float>(i, 0) << endl;
		}
		cout << endl;

		cv::Ptr< cv::ml::DTrees > Mytree = cv::ml::DTrees::create();

		//  Mytree->setMinSampleCount(1);
		Mytree->setCVFolds(1);
		Mytree->setMaxDepth(3);
		Mytree->setRegressionAccuracy(.01f);

		Mytree->train(X, cv::ml::ROW_SAMPLE, ff); // тренирует дерево на TrainData


		int NN = 150;
		Mat X_preduct(NN, 1, CV_32FC1);
		Mat results;
		float hh = 3.14 / NN;
		for (int i = 0; i < NN; i++)
		{
			X_preduct.at<float>(i, 0) = i * hh;
		}

		Mytree->predict(X_preduct, results);

		for (int i = 0; i < results.rows; i++) {
			for (int j = 0; j < results.cols; j++)
				cout << X_preduct.at<float>(i, 0) << " " << results.at<float>(i, j);
			cout << endl;
		}
}*/
