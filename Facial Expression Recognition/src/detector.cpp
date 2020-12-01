#include <string>
#include <iostream>
#include <filesystem>

#include <Eigen/Dense>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

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
	//svm = ml::SVM::load("ml/svm.yml");
	//ann = ml::ANN_MLP::load("ml/ann.yml");
}

void detector::init() {

	//Inicialização do descritor hog
	hog = new HOGDescriptor(Size(128, 128), Size(32, 32), Size(16, 16), Size(16, 16), 9, 1);

	//Inicialização do SVM
	svm = ml::SVM::create();

	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::INTER);
	svm->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-5));

	//Inicialização da rede neural mlp
	ann = ml::ANN_MLP::create();
	Mat_<int> layers(4, 1);

	int nfeatures = 63 * 4 * 2 + 7*7*4*9;
	layers(0) = nfeatures;
	layers(1) = nclasses * 32;
	layers(2) = nclasses * 16;
	layers(3) = nclasses;

	ann->setLayerSizes(layers);
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
	ann->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 1e-3));
	ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
}

void detector::train() {

	init();

	std::string path = "../datas/images/DBs/Yale/train";

	Mat trainData; // entrada para trinamento
	Mat trainClassesMLP, trainClassesSVM; // saída esperada

	int cont = 0;
	for (const auto& entry : fs::directory_iterator(path)) {
		std::string file = entry.path().string();
		
		Mat img = imread(file);
		DLIBImage dlibImg(img);
		
		if (img.empty()) {
			std::cout << "---PROBLEM" << file << std::endl;
			continue;
		}

		//----------------------//
		//PARA O CK DATABASE
		//DLIBRects dlibRects;
		//dlibRects.push_back(dlib::rectangle(0, 0, img.cols, img.rows));
		//PARA OUTROS DATABASES
		DLIBRects dlibRects = faceDetector(dlibImg);
		//---------------------//


		if (dlibRects.empty()) {
			std::cout << "---PROBLEM" << file << std::endl;
			continue;
		}

		for (int i = 0; i < dlibRects.size(); i++){
			Mat geometricFeatures; 
			full_object_detection landmarks;
			getLandmarks(landmarks, dlibImg, dlibRects[i]);
			expression facialExpression = expression(img, landmarks);
			geometricFeatures = facialExpression.getFeatures();

			std::vector<float> descriptor;
			Rect r = dlibRectToOpenCV(dlibRects[i]);
			Mat resizedImg;
			cv::resize(img, resizedImg, Size(128, 128));
			hog->compute(resizedImg, descriptor);
			Mat apparenceFeatures(1, descriptor.size(), cv::DataType<float>::type, descriptor.data());


			Mat features;
			hconcat(apparenceFeatures, geometricFeatures, features);
			trainData.push_back(features);

			trainClassesMLP.push_back(Mat::zeros(1, nclasses, CV_32F));
			trainClassesSVM.push_back(Mat::zeros(1, 1, CV_32S));

			if (file.find("happy") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 0) = 1;
				trainClassesSVM.at<int>(cont, 0) = 0;
			}
			else if (file.find("neutral") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 1) = 1;
				trainClassesSVM.at<int>(cont, 0) = 1;
			}
			else if (file.find("sad") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 2) = 1;
				trainClassesSVM.at<int>(cont, 0) = 2;
			}
			else if (file.find("sleepy") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 3) = 1;
				trainClassesSVM.at<int>(cont, 0) = 3;
			}
			else if (file.find("surprised") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 4) = 1;
				trainClassesSVM.at<int>(cont, 0) = 4;
			}else if (file.find("wink") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 5) = 1;
				trainClassesSVM.at<int>(cont, 0) = 5;
			}
			
		}
		
		cont++;
	}

	svm->train(trainData, ml::ROW_SAMPLE, trainClassesSVM);
	ann->train(trainData, ml::ROW_SAMPLE, trainClassesMLP);
	
	//saveTrain();
}

void detector::saveTrain() {
	svm->save("ml/svm.yml");
	ann->save("ml/ann.yml");
}

void detector::test() {

	std::string path = "../datas/images/DBs/Yale/test";
	Mat testData;
	Mat testLabels;
	
	for (const auto& entry : fs::directory_iterator(path)) {
		std::string file = entry.path().string();

		Mat img = imread(file);
		DLIBImage dlibImg(img);
		
		if (img.empty()) {
			std::cout << "---PROBLEM" << file << std::endl;
			continue;
		}

		DLIBRects dlibRects = faceDetector(dlibImg);
		if (dlibRects.empty()) {
			std::cout << "---PROBLEM" << file << std::endl;
			continue;
		}

		for (int i = 0; i < dlibRects.size(); i++) {

			Mat geometricFeatures;
			full_object_detection landmarks;
			getLandmarks(landmarks, dlibImg, dlibRects[i]);
			expression facialExpression = expression(img, landmarks);
			geometricFeatures = facialExpression.getFeatures();

			std::vector<float> descriptor;
			Rect r = dlibRectToOpenCV(dlibRects[i]);
			Mat resizedImg;
			cv::resize(img, resizedImg, Size(128, 128));
			hog->compute(resizedImg, descriptor);
			Mat apparenceFeatures(1, descriptor.size(), cv::DataType<float>::type, descriptor.data());

			Mat features;
			hconcat(apparenceFeatures, geometricFeatures, features);
			testData.push_back(features);

			if (file.find("happy") != std::string::npos) {
				testLabels.push_back(0);
			}
			else if (file.find("neutral") != std::string::npos) {
				testLabels.push_back(1);
			}
			else if (file.find("sad") != std::string::npos) {
				testLabels.push_back(2);
			}
			else if (file.find("sleepy") != std::string::npos) {
				testLabels.push_back(3);
			}
			else if (file.find("surprised") != std::string::npos) {
				testLabels.push_back(4);
			}
			else if (file.find("wink") != std::string::npos) {
				testLabels.push_back(5);
			}
		}
	}

	int predMLP, predSVM;
	int truthMLP, truthSVM;
	Mat resultMLP, resultSVM;
	Mat confusionMLP(nclasses, nclasses, CV_32S, Scalar(0));
	Mat confusionSVM(nclasses, nclasses, CV_32S, Scalar(0));
	for (int i = 0; i < testData.rows; i++) {
		predMLP = ann->predict(testData.row(i), resultMLP);
		predSVM = svm->predict(testData.row(i), resultSVM);
		predSVM = (int) resultSVM.at<float>(0,0);
		truthMLP = testLabels.at<int>(i);
		truthSVM = testLabels.at<int>(i);
		confusionMLP.at<int>(predMLP, truthMLP)++;
		confusionSVM.at<int>(predSVM, truthSVM)++;

		//cout << "cont: " << i << endl;
		//std::cout << "pred: " << predMLP << "     pred SVM: " << predSVM << std::endl;
		//cout << "truth: " << truthMLP << "     truth SVM" << truthSVM << endl;
		//cout << "confusion: \n" << confusionMLP << endl << endl;
		//cout << "confusionsvm: \n" << confusionSVM << endl << endl;
		//cout  << resultMLP << endl << endl;
		//cout << resultSVM << endl << endl;
	}


	Mat correct = confusionMLP.diag();
	float accuracyMLP = sum(correct)[0] / sum(confusionMLP)[0];
	std::cerr << "accuracy: " << accuracyMLP << std::endl;
	std::cerr << "confusion:\n " << confusionMLP << std::endl;

	Mat correctSVM = confusionSVM.diag();
	float accuracySVM = sum(correctSVM)[0] / sum(confusionSVM)[0];
	std::cerr << "accuracySVM: " << accuracySVM << std::endl;
	std::cerr << "confusionSVM:\n " << confusionSVM << std::endl;
}

cv::Rect detector::dlibRectToOpenCV(dlib::rectangle r)
{
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right(), r.bottom()));
}

void detector::getLandmarks(full_object_detection &flNormalized, DLIBImage dlibImg,dlib::rectangle r) 
{
	full_object_detection faceLandmarks = landmarkDetector(dlibImg, r);
	chip_details chip = get_face_chip_details(faceLandmarks, 128);
	flNormalized = map_det_to_chip(faceLandmarks, chip);

}
