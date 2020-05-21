#include "commons.h"
#include <time.h>
#include <iomanip>
#include "stdlib.h"
#include "math.h"
#include "canny.h"

#include "utils.h"
#include <iostream>


cv::Mat cscThresholding (const cv::Mat& rgbInput) {
    cv::Mat hsvImg, gray, yellowHueRange, whiteHueRange, mask;
	cvtColor(rgbInput, hsvImg, COLOR_BGR2HSV);
    cvtColor(rgbInput, gray, COLOR_BGR2GRAY);
    
    auto lowerYellow = Scalar(20, 100, 100);
    auto upperYellow = Scalar(30, 255, 255);

    auto lowerWhite = Scalar(120, 120, 120);
    auto upperWhite = Scalar(255, 255, 255);

	inRange(hsvImg, lowerYellow, upperYellow, yellowHueRange);
	inRange(rgbInput, lowerWhite, upperWhite, whiteHueRange);
	bitwise_or(yellowHueRange, whiteHueRange, mask);
	bitwise_and(gray, mask, gray);

	return gray;
}


Mat regionOfInterest(Mat& img) {
	Mat mask(img.rows, img.cols, CV_8UC1, Scalar(0));

	vector<Point> vertices;
	vertices.push_back(Point(img.cols / 9, img.rows));
	vertices.push_back(Point(img.cols - (img.cols / 9), img.rows));
	vertices.push_back(Point((img.cols / 2) + (img.cols / 8), (img.rows / 2) + (img.rows / 10)));
	vertices.push_back(Point((img.cols / 2) - (img.cols / 8), (img.rows / 2) + (img.rows / 10)));

	// Create Polygon from vertices
	vector<Point> ROI_Poly;
	approxPolyDP(vertices, ROI_Poly, 1.0, true);

	// Fill polygon white
	fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0);
	bitwise_and(img, mask, img);

	return img;
}


void getImageFromVector(const float* buffer,
                        cv::Mat& out,
                        int width,
                        int height) {
    for(int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
            uint8_t pixel;
            pixel = (uint8_t)(buffer[j*width + i] * 255);
            out.at<uint8_t>(j, i) = pixel;
        }
    }
}


cv::Mat processCanny (const cv::Mat& gray) {
	float *hostInputImageData, *hostOutputImageBuffer;
    hostInputImageData    = (float*)malloc(sizeof(float) * gray.rows * gray.cols);
	hostOutputImageBuffer = (float*)malloc(sizeof(float) * gray.rows * gray.cols);
    cv::Mat out = gray;

    for(int i=0; i<gray.cols; i++) {
        for (int j=0; j<gray.rows; j++) {
            uint8_t pix = gray.at<uint8_t>(j, i);
            hostInputImageData[j*gray.cols + i] = (float)(pix / 255.0f);
        }
    }

    CannyEdgeDevice* cannyContext = new CannyEdgeDevice(hostInputImageData, gray.cols, gray.rows);

	cannyContext->performGaussianFiltering();
    cannyContext->performImageGradientX();
    cannyContext->performImageGradientY();
    cannyContext->computeMagnitude();
    cannyContext->nonMaxSuppression();
    cannyContext->computeCannyThresholds();
    cannyContext->highHysterisisThresholding();
    cannyContext->lowHysterisisThresholding();

    float *cannyLowThresholdOutput = cannyContext->getD_LowThreshold();

    cudaMemcpy(hostOutputImageBuffer, cannyLowThresholdOutput, sizeof(float) * gray.rows * gray.cols, cudaMemcpyDeviceToHost);
    getImageFromVector(hostOutputImageBuffer, out, gray.cols, gray.rows);

    free(hostOutputImageBuffer);
    free(hostInputImageData);

    delete cannyContext;

    return out;
}
