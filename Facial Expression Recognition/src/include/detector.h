#pragma once

class detector
{
private:
	int nclasses = 2; //Número de saídas
	shape_predictor landmarkDetector;
	std::vector<full_object_detection> faceLandmarks;
	frontal_face_detector faceDetector;
	expression facialExpression;

	Ptr<ml::ANN_MLP> ann;

public:
	detector();
	void setExpression(expression &facialExpression_);
	void getLandmarks(Mat image);
	void train();
	void readWeights();
};

