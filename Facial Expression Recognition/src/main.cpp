#include <iostream>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace cv;
using namespace dlib;

#include "include/splines.h"
#include "include/ExpressionDetector.h"

int main() {
	Mat frame;
	VideoCapture cam("../files/video/girl2.mp4");

	frontal_face_detector detector;
	full_object_detection faceLandmarks;
	detector = get_frontal_face_detector();
	std::vector<dlib::rectangle> faceRects;
	
	shape_predictor landmarkDetector;
	deserialize("../files/data/predictor_face_landmarks.dat") >> landmarkDetector;

	while (1) {
		cam >> frame;
		if (frame.empty()) break;
			

			cv_image<bgr_pixel> dlib_frame(frame);
			faceRects = detector(dlib_frame);
			
			for (int i = 0; i < faceRects.size(); i++) {
				faceLandmarks = landmarkDetector(dlib_frame, faceRects[i]);
				ExpressionDetector expression_detector(frame, faceLandmarks);
			}

			cv::imshow("VIDEO", frame);
			waitKey(1);
	}

	return 0;
}