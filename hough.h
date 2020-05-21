#include "commons.h"
#include "utils.h"

#define THETA_STEP_SIZE 0.1
#define RHO_STEP_SIZE 2
#define THRESHOLD 125
#define THETA_A 45.0
#define THETA_B 135.0
#define THETA_VARIATION 16.0
#define MAX_NUM_LINES 10

class Line {
private:
    double theta;
    double rho;
    
public:
    Line(double theta, double rho);
    /** Calculates y value of line based on given x */
    double getY(double x);
    /** Calculates x value of line based on given y */
    double getX(double y);
};


class HoughTransformDevice {
	int nRows;
	int nCols;

	int frameSize;
	int *lines;
	int *d_lines;
	int lineCounter;
	int *d_lineCounter;
	uchar *d_frame;
	int *d_accumulator;
	dim3 houghBlockDim;
	dim3 houghGridDim;
	dim3 findLinesBlockDim;
	dim3 findLinesGridDim;

	public:
		HoughTransformDevice(int nRows, int nCols);
		~HoughTransformDevice();

		void processFrame(cv::Mat& frame, std::vector<Line>& lines);
};
