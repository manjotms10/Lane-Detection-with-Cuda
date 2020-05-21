#include "commons.h"
#include "hough.h"
#include "utils.h"
#include <time.h>
#include <iomanip>

void detectLanes(VideoCapture inputVideo, VideoWriter outputVideo);
void drawLines(Mat &frame, vector<Line> lines);
cv::Mat plotAccumulator(int nRows, int nCols, int *accumulator);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "usage LaneDetection inputVideo outputVideo" << endl << endl;
        cout << "Positional Arguments:" << endl;
        cout << " inputVideo    Input video for which lanes are detected" << endl;
        cout << " outputVideo   Name of resulting output video (created by the program)" << endl;
        return 1;
    }

    // Read input video
    VideoCapture inputVideo(argv[1]);

    int frameWidth  = inputVideo.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = inputVideo.get(CAP_PROP_FRAME_HEIGHT);

    if (!inputVideo.isOpened()) {
        cout << "Unable to open video" << endl;
        return -1;
    }

    VideoWriter outputVideo(argv[2], VideoWriter::fourcc('M','J','P','G'), 30,
                            cv::Size(frameWidth, frameHeight), true);

    detectLanes(inputVideo, outputVideo);

    return 0;
}


void detectLanes(VideoCapture inputVideo, VideoWriter outputVideo) {
    cv::Mat frame, preProcessedFrame;
    std::vector<Line> lines;

    int frameCount = 0;

    clock_t readTime = 0;
	clock_t prepTime = 0;
	clock_t houghTime = 0;
	clock_t writeTime = 0;
    clock_t totalTime = -clock();

    int frameWidth = inputVideo.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = inputVideo.get(CAP_PROP_FRAME_HEIGHT);

    HoughTransformDevice *handle = new HoughTransformDevice(frameWidth, frameHeight);

    std::cout << "Processing video using CUDA ..." << std::endl;

	for( ; ; ) {
        // Read next frame
        readTime -= clock();
        inputVideo >> frame;
        readTime += clock();
        if(frame.empty()) {
            break;
        }

        prepTime -= clock();
        preProcessedFrame = cscThresholding(frame);
        preProcessedFrame = processCanny(preProcessedFrame);
        preProcessedFrame = regionOfInterest(preProcessedFrame);
        prepTime += clock();

        houghTime -= clock();
        lines.clear();
        handle->processFrame(preProcessedFrame, lines);
        houghTime += clock();

        // Draw lines to frame and write to output video
        writeTime -= clock();
        drawLines(frame, lines);

        // Write frame to the output video.
        outputVideo << frame;
        writeTime += clock();
        frameCount++;
    }

    delete handle;

    totalTime += clock();
	cout << "Read\tPrep\tHough\tWrite\tTotal\tFPS" << endl;
	cout << setprecision (4)<<(((float) readTime) / CLOCKS_PER_SEC) << "\t"
         << (((float) prepTime) / CLOCKS_PER_SEC) << "\t"
		 << (((float) houghTime) / CLOCKS_PER_SEC) << "\t"
		 << (((float) writeTime) / CLOCKS_PER_SEC) << "\t"
    	 << (((float) totalTime) / CLOCKS_PER_SEC) << "\t"
         << (((float) frameCount) * CLOCKS_PER_SEC / totalTime) << endl;
}


void drawLines(Mat &frame, vector<Line> lines) {
    for (int i = 0; i < lines.size(); i++) {
        int y1 = frame.rows;
        int y2 = (frame.rows / 2) + (frame.rows / 10);
        int x1 = (int) lines[i].getX(y1);
        int x2 = (int) lines[i].getX(y2);

        line(frame, Point(x1, y1), Point(x2, y2), Scalar(255), 5, 8, 0);
    }
}


cv::Mat plotAccumulator(int nRows, int nCols, int *accumulator) {
	cv::Mat plotImg(nRows, nCols, CV_8UC1, Scalar(0));
	for (int i = 0; i < nRows; i++) {
  		for (int j = 0; j < nCols; j++) {
			plotImg.at<uchar>(i, j) = min(accumulator[(i * nCols) + j] * 4, 255);
  		}
  	}

    return plotImg;
}
