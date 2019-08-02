
/*
김진엽
2019.08.03 
openCV 얼굴검출 및 모자이크 (오픈소스)
*/

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/core_c.h>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name;
CascadeClassifier face_cascade;
String window_name = "Face detection";

//모자이크[펌]
void drawMosaicRectangle(cv::Mat frame, cv::Rect face) {
	int cnts = 0;
	int mb = 9;
	int wPoint = 0;
	int hPoint = 0;
	int xStartPoint = 0;
	int yStartPoint = 0;
	double R = 0;
	double G = 0;
	double B = 0;

	for (int i = 0; i < face.height / mb; i++) {
		for (int j = 0; j < face.width / mb; j++) {
			cnts = 0;
			B = 0;
			G = 0;
			R = 0;
			xStartPoint = face.x + (j * mb);
			yStartPoint = face.y + (i * mb);

			// 이미지의 픽셀 값의 r, g, b 값의 각각 합을 구함
			for (int mbY = yStartPoint; mbY < yStartPoint + mb; mbY++) {
				for (int mbX = xStartPoint; mbX < xStartPoint + mb; mbX++) {
					wPoint = mbX;
					hPoint = mbY;

					if (mbX >= frame.cols) {
						wPoint = frame.cols - 1;
					}
					if (mbY >= frame.rows) {
						hPoint = frame.rows - 1;
					}

					cv::Vec3b color = frame.at<cv::Vec3b>(hPoint, wPoint);
					B += color.val[0];
					G += color.val[1];
					R += color.val[2];
					cnts++;
				}
			}

			// r, g, b 값의 평균 산출
			B /= cnts;
			G /= cnts;
			R /= cnts;

			// 모자이크 색상 생성
			cv::Scalar color;
			color.val[0] = B;
			color.val[1] = G;
			color.val[2] = R;

			// 프레임에 모자이크 이미지 삽입
			cv::rectangle(
				frame,
				cvPoint(xStartPoint, yStartPoint),
				cvPoint(xStartPoint + mb, yStartPoint + mb),
				color,
				cv::FILLED,
				8,
				0
			);
		}
	}
}

/** @function main */
int main(int argc, const char** argv)
{
	face_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };

	VideoCapture cam(0);
	Mat frame;

	//cam.set(CV_CAP_PROP_FRAME_WIDTH, CAM_WIDTH);
	//cam.set(CV_CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT);

	if (!cam.isOpened()) { printf("--(!)Error opening video cam\n"); return -1; }

	while (cam.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No camd frame -- Break!");
			break;
		}

		detectAndDisplay(frame);
		char c = (char)waitKey(10);
		if (c == 27) { break; }
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(200, 200));

	for (size_t i = 0; i < faces.size(); i++)
	{
		drawMosaicRectangle(frame, faces[i]);
		cv::Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		cv::Point tr(faces[i].x, faces[i].y);
		cv::rectangle(frame, lb, tr, cv::Scalar(0, 0, 255), 1/*프레임굵기*/, 4, 0);
	}

	imshow(window_name, frame);
}