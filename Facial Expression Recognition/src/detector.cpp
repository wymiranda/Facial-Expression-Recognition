#include <string>
#include <iostream>
#include <filesystem>

#include <Eigen/Dense>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace dlib;
namespace fs = std::filesystem;

#include "include/splines.h"
#include "include/expression.h"
#include "include/detector.h"

detector::detector() {
	deserialize("../files/data/predictor_face_landmarks.dat") >> landmarkDetector;
	faceDetector = get_frontal_face_detector();
}



void detector::setExpression(expression &facialExpression_) {
	facialExpression = facialExpression_;
}

void detector::train() {

	std::string path = "../files/images";
	
	ann = ml::ANN_MLP::create();
	Mat_<int> layers(4, 1);

	int nfeatures = 63 * 4 * 2;
	layers(0) = nfeatures;
	layers(1) = nclasses * 20;
	layers(2) = nclasses * 10;
	layers(3) = nclasses;

	ann->setLayerSizes(layers);
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
	ann->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.0001));
	ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);

	using std::filesystem::directory_iterator;
	int nfiles = std::distance(directory_iterator(path), directory_iterator{});

	Mat trainData; // entrada para trinamento
	Mat trainClasses; // saída esperada
	int cont = 0;
	for (const auto& entry : fs::directory_iterator(path)) {
		std::string file = entry.path().string();
		Mat img = imread(file);
		
		getLandmarks(img);
		if (faceLandmarks.size() == 0) continue;
		facialExpression = expression(img, faceLandmarks[0]);
		if (cont == 0) {
			trainData = facialExpression.getFeatures();
			trainClasses = Mat::zeros(1, nclasses, CV_32FC1);
			if (file.find("normal") != std::string::npos) {
				trainClasses.at<float>(cont, 0) = 1;
			}if (file.find("happy") != std::string::npos) {
				trainClasses.at<float>(cont, 1) = 1;
			}
		}
		else {
			vconcat(trainData, facialExpression.getFeatures(), trainData);
			vconcat(trainClasses, Mat::zeros(1, nclasses, CV_32FC1), trainClasses);
			if (file.find("normal") != std::string::npos) {
				trainClasses.at<float>(cont, 0) = 1;
			}if (file.find("happy") != std::string::npos) {
				trainClasses.at<float>(cont, 1) = 1;
			}
		}
		cont++;
	}

	ann->train(trainData, ml::ROW_SAMPLE, trainClasses);



	Mat results;
	Mat teste = imread("../files/images/teste (1).jpg");
	getLandmarks(teste);
	expression express(teste, faceLandmarks[0]);
	float pred = ann->predict(express.getFeatures(), results);
	std::cout << "mulher sorrindo: " << pred << std::endl << results << std::endl;

	 teste = imread("../files/images/teste (2).jpg");
	 getLandmarks(teste);
	 express = expression(teste, faceLandmarks[0]);
	 pred = ann->predict(express.getFeatures(), results);
	 std::cout << "mulher seria: " << pred << std::endl << results << std::endl;

	 teste = imread("../files/images/teste (3).jpg");
	 getLandmarks(teste);
	 express = expression(teste, faceLandmarks[0]);
	 pred = ann->predict(express.getFeatures(), results);
	 std::cout << "mulher sorrindo: " << pred << std::endl << results << std::endl;

	 teste = imread("../files/images/teste (4).jpg");
	 getLandmarks(teste);
	 express = expression(teste, faceLandmarks[0]);
	 pred = ann->predict(express.getFeatures(), results);
	 std::cout << "homem serio: " << pred << std::endl << results << std::endl;


	//std::cout << "Voce gostaria de sobrescrever o ultimo o treinamento? (s/n): ";
	//std::string reset;
	//std::cin >> reset;
	//if (reset == "s") {
	//} o que acontece???????????????????
}

void detector::readWeights() {

}

void detector::getLandmarks(Mat image) {
	cv_image<bgr_pixel> dlib_image(image);
	std::vector<dlib::rectangle> faceRects;
	faceRects = faceDetector(dlib_image);

	faceLandmarks.clear();
	for (int i = 0; i < faceRects.size(); i++) {
		full_object_detection shape = landmarkDetector(dlib_image, faceRects[i]);
		chip_details chip = get_face_chip_details(shape, 100);
		faceLandmarks.push_back(map_det_to_chip(shape, chip));
		
	}
}
