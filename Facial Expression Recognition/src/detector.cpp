#include <string>
#include <iostream>
#include <filesystem>

#include <Eigen/Dense>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dlib;
namespace fs = std::filesystem;

#include "include/splines.h"
#include "include/expression.h"
#include "include/detector.h"

detector::detector() {
	deserialize("../datas/face_predictor/predictor_face_landmarks.dat") >> landmarkDetector;
	faceDetector = get_frontal_face_detector();
	//SVM
	//ann = ml::ANN_MLP::load("../datas/ann/ann.yml");
}

void detector::setExpression(expression &facialExpression_) {
	facialExpression = facialExpression_;
}

void detector::train() {

	std::string path = "../datas/images/train";
	
	ann = ml::ANN_MLP::create();
	Mat_<int> layers(4, 1);

	int nfeatures = 63 * 4 * 2;
	layers(0) = nfeatures;
	layers(1) = nclasses * 16;
	layers(2) = nclasses * 8;
	layers(3) = nclasses;

	svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::INTER);
	svm->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.001));

	ann->setLayerSizes(layers);
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
	ann->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.001));
	ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.000001);

	using std::filesystem::directory_iterator;
	int nfiles = std::distance(directory_iterator(path), directory_iterator{});

	Mat trainData; // entrada para trinamento
	Mat trainClasses, trainClassesSVM; // saída esperada
	int cont = 0;
	for (const auto& entry : fs::directory_iterator(path)) {
		std::string file = entry.path().string();
		Mat img = imread(file);
		
		if (img.empty()) {
			std::cout << "---PROBLEM" << file << std::endl;
			continue;
		}

		getLandmarks(img);
		if (faceLandmarks.size() == 0) {
			std::cout << "---PROBLEM" << file << std::endl;
			continue;
		}
		facialExpression = expression(img, faceLandmarks[0]);

		if (cont == 0) {
			trainData = facialExpression.getFeatures();
			
			trainClasses = Mat::zeros(1, nclasses, CV_32F);
			trainClassesSVM = Mat::zeros(1, 1, CV_32S);

			if (file.find("anger") != std::string::npos) {
				trainClasses.at<float>(cont, 0) = 1;
				trainClassesSVM.at<int>(0, 0) = 0;
			}else if (file.find("happy") != std::string::npos) {
				trainClasses.at<float>(cont, 1) = 1;
				trainClassesSVM.at<int>(0, 1) = 1;
			}
			else if (file.find("neutral") != std::string::npos) {
				trainClasses.at<float>(cont, 2) = 1;
				trainClassesSVM.at<int>(0, 2) = 2;
			}
			else if (file.find("sad") != std::string::npos) {
				trainClasses.at<float>(cont, 3) = 1;
				trainClassesSVM.at<int>(0, 0) = 3;
			}
			else if (file.find("surprised") != std::string::npos) {
				trainClasses.at<float>(cont, 4) = 1;
				trainClassesSVM.at<int>(0, 0) = 4;
			}
		}
		else {
			vconcat(trainData, facialExpression.getFeatures(), trainData);
			vconcat(trainClasses, Mat::zeros(1, nclasses, CV_32F), trainClasses);
			hconcat(trainClassesSVM, Mat::zeros(1, 1, CV_32S), trainClassesSVM);

			if (file.find("anger") != std::string::npos) {
				trainClasses.at<float>(cont, 0) = 1;
				trainClassesSVM.at<int>(0, cont) = 0;
			}
			else if (file.find("happy") != std::string::npos) {
				trainClasses.at<float>(cont, 1) = 1;
				trainClassesSVM.at<int>(0, cont) = 1;
			}
			else if (file.find("neutral") != std::string::npos) {
				trainClasses.at<float>(cont, 2) = 1;
				trainClassesSVM.at<int>(0, cont) = 2;
			}
			else if (file.find("sad") != std::string::npos) {
				trainClasses.at<float>(cont, 3) = 1;
				trainClassesSVM.at<int>(0, cont) = 3;
			}
			else if (file.find("surprised") != std::string::npos) {
				trainClasses.at<float>(cont, 4) = 1;
				trainClassesSVM.at<int>(0, cont) = 4;
			}
		}
		
	cont++;
	}

	ann->train(trainData, ml::ROW_SAMPLE, trainClasses);
	svm->train(trainData, ml::ROW_SAMPLE, trainClassesSVM);
	//saveTrain();
}

void detector::saveTrain() {
	//std::cout << "Voce gostaria de sobrescrever o ultimo o treinamento? (s/n): ";
	//std::string reset;
	//std::cin >> reset;
	//if (reset == "s") {
		 // or xml
		ann->save("../datas/ann/ann.yml"); // don't think too much about the deref, it casts to a FileNode
	//}
}

void detector::test() {

	std::string path = "../datas/images/test";
	Mat testData;
	Mat testLabels;
	int cont = 0;
	for (const auto& entry : fs::directory_iterator(path)) {
		std::string file = entry.path().string();
		Mat img = imread(file);

		if (img.empty()) continue;

		getLandmarks(img);
		if (faceLandmarks.size() == 0) continue;
		facialExpression = expression(img, faceLandmarks[0]);

		if (cont == 0) {
			testData = facialExpression.getFeatures();
			testLabels = Mat::zeros(1, 1, CV_32S);

			if (file.find("anger") != std::string::npos) {
				testLabels.at<int>(cont, 0) = 0;
			}else if (file.find("happy") != std::string::npos) {
				testLabels.at<int>(cont, 0) = 1;
			}else if (file.find("neutral") != std::string::npos) {
				testLabels.at<int>(cont, 0) = 2;
			}else if (file.find("sad") != std::string::npos) {
				testLabels.at<int>(cont, 0) = 3;
			}else if (file.find("surprised") != std::string::npos) {
				testLabels.at<int>(cont, 0) = 4;
			}
		}
		else {
			vconcat(testData, facialExpression.getFeatures(), testData);
			vconcat(testLabels, Mat::zeros(1, 1, CV_32S), testLabels);

			if (file.find("anger") != std::string::npos) {
				testLabels.at<int>(cont, 0) = 0;
			}else if (file.find("happy") != std::string::npos) {
				testLabels.at<int>(cont, 0) = 1;
			}else if (file.find("neutral") != std::string::npos) {
				testLabels.at<int>(cont, 0) = 2;
			}else if (file.find("sad") != std::string::npos) {
				testLabels.at<int>(cont, 0) = 3;
			}else if (file.find("surprised") != std::string::npos) {
				testLabels.at<int>(cont, 0) = 4;
			}
		}


		cont++;
	}
	int pred, predSVM;
	int truth, tSVM;
	Mat result, rSVM(1,1,CV_32F);
	Mat confusion(nclasses, nclasses, CV_32S, Scalar(0));
	Mat confusionSVM(nclasses, nclasses, CV_32S, Scalar(0));
	for (int i = 0; i < testData.rows; i++) {
		pred = ann->predict(testData.row(i), result);
		predSVM = svm->predict(testData.row(i), rSVM);
		predSVM = (int) rSVM.at<float>(0,0);
		truth = testLabels.at<int>(i);
		tSVM = testLabels.at<int>(i);
		confusion.at<int>(pred, truth)++;
		confusionSVM.at<int>(predSVM, tSVM)++;

		cout << "cont: " << i << endl;
		std::cout << "pred: " << pred << "     pred SVM: " << predSVM << std::endl;
		cout << "truth: " << truth << "     truth SVM" << tSVM << endl;
		cout << "confusion: \n" << confusion << endl << endl;
		cout << "confusionsvm: \n" << confusionSVM << endl << endl;
		cout  << result << endl << endl;
		cout << rSVM << endl << endl;
	}


	Mat correct = confusion.diag();
	float accuracy = sum(correct)[0] / sum(confusion)[0];
	std::cerr << "accuracy: " << accuracy << std::endl;
	std::cerr << "confusion:\n " << confusion << std::endl;

	Mat correctSVM = confusionSVM.diag();
	float accuracySVM = sum(correctSVM)[0] / sum(confusionSVM)[0];
	std::cerr << "accuracySVM: " << accuracySVM << std::endl;
	std::cerr << "confusionSVM:\n " << confusionSVM << std::endl;
}

void detector::getLandmarks(Mat image) {
	cv_image<bgr_pixel> dlib_image(image);
	std::vector<dlib::rectangle> faceRects;
	faceRects = faceDetector(dlib_image);

	faceLandmarks.clear();
	for (int i = 0; i < faceRects.size(); i++) {
		full_object_detection shape = landmarkDetector(dlib_image, faceRects[i]);
		chip_details chip = get_face_chip_details(shape, 200);
		faceLandmarks.push_back(map_det_to_chip(shape, chip));
		//faceLandmarks.push_back(shape);
		
	}
}
