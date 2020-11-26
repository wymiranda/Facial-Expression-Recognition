#pragma once

class detector
{
private:
	int nclasses = 5; //Número de saídas
	shape_predictor landmarkDetector;
	std::vector<full_object_detection> faceLandmarks;
	frontal_face_detector faceDetector;
	expression facialExpression;

	Ptr<ml::ANN_MLP> ann;
	Ptr<ml::SVM> svm;

	void setExpression(expression& facialExpression_);
	void getLandmarks(Mat image);
	void saveTrain();
public:
	detector();
	void test();
	void train();
};

