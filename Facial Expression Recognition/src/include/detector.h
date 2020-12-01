#pragma once

class detector
{
private:
	int nclasses = 6; //N�mero de sa�das
	shape_predictor landmarkDetector;
	frontal_face_detector faceDetector;
	typedef std::vector<dlib::rectangle> DLIBRects;
	typedef cv_image<bgr_pixel> DLIBImage;

	HOGDescriptor* hog;
	Ptr<ml::ANN_MLP> ann;
	Ptr<ml::SVM> svm;

	void init();
	Rect dlibRectToOpenCV(dlib::rectangle r);
	void getLandmarks(full_object_detection& flNormalized, DLIBImage dlibImg, dlib::rectangle r);
	void saveTrain();
public:
	detector();
	void test();
	void train();
};

