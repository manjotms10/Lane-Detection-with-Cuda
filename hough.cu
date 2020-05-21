#include "hough.h"
#include "commons.h"

__host__ __device__ double calcRho(double x, double y, double theta) {
    double thetaRadian = (theta * PI) / 180.0;

    return x * cos(thetaRadian) + y * sin(thetaRadian);
}

__host__ __device__ int index(int nRows, int nCols, int rho, double theta) {
    return ((rho / RHO_STEP_SIZE) + (nRows / 2)) * nCols +
            (int) ((theta - (THETA_A-THETA_VARIATION)) / THETA_STEP_SIZE + 0.5);
}

__host__ __device__ bool isLocalMaximum(int i, int j, int nRows, int nCols, int *accumulator) {
   for (int i_delta = -50; i_delta <= 50; i_delta++) {
       for (int j_delta = -50; j_delta <= 50; j_delta++) {
           if (i + i_delta > 0 && i + i_delta < nRows && j + j_delta > 0 && j + j_delta < nCols &&
               accumulator[(i + i_delta) * nCols + j + j_delta] > accumulator[i * nCols + j]) {
               return false;
           }
       }
   }

   return true;
}

__global__ void houghKernel(int frameWidth, int frameHeight, unsigned char* frame, int nRows, int nCols, int *accumulator) {
    int i = blockIdx.x * blockDim.y + threadIdx.y;
    int j = blockIdx.y * blockDim.z + threadIdx.z;
    double theta;
    int rho;

    if(i < frameHeight && j < frameWidth && ((int) frame[(i * frameWidth) + j]) != 0) {

        // thetas of interest will be close to 45 and close to 135 (vertical lines)
        // we are doing 2 thetas at a time, 1 for each theta of Interest
        // we use thetas varying 15 degrees more and less
        for(int k = threadIdx.x * (1 / THETA_STEP_SIZE); k < (threadIdx.x + 1) * (1 / THETA_STEP_SIZE); k++) {
            theta = THETA_A-THETA_VARIATION + ((double)k*THETA_STEP_SIZE);
            rho = calcRho(j, i, theta);
            atomicAdd(&accumulator[index(nRows, nCols, rho, theta)], 1);

            theta = THETA_B-THETA_VARIATION + ((double)k*THETA_STEP_SIZE);
            rho = calcRho(j, i, theta);
            atomicAdd(&accumulator[index(nRows, nCols, rho, theta)], 1);
        }
    }
}

__global__ void findLinesKernel(int nRows, int nCols, int *accumulator, int *lines, int *lineCounter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (accumulator[i * nCols + j] >= THRESHOLD && isLocalMaximum(i, j, nRows, nCols, accumulator)) {
        int insertPt = atomicAdd(lineCounter, 2);
        if (insertPt + 1 < 2 * MAX_NUM_LINES) {
            lines[insertPt] = THETA_A-THETA_VARIATION + (j * THETA_STEP_SIZE);
            lines[insertPt + 1] = (i - (nRows / 2)) * RHO_STEP_SIZE;
        }
    }
}

Line::Line(double theta, double rho) {
    this->theta = theta;
    this->rho = rho;
}

/** Calculates y value of line based on given x */
double Line::getY(double x) {
    double thetaRadian = (theta * PI) / 180.0;
    return (rho  - x * cos(thetaRadian)) / sin(thetaRadian);
}

/** Calculates x value of line based on given y */
double Line::getX(double y) {
    double thetaRadian = (theta * PI) / 180.0;
    return (rho - y * sin(thetaRadian)) / cos(thetaRadian);
}


void HoughTransformDevice::processFrame(cv::Mat& frame, std::vector<Line>& outputLines) {
    cudaMemcpy(d_frame, frame.ptr(), frameSize, cudaMemcpyHostToDevice);
    cudaMemset(d_accumulator, 0, nRows * nCols * sizeof(int));

    houghKernel <<<houghGridDim, houghBlockDim>>> (frame.cols, frame.rows, d_frame, nRows, nCols, d_accumulator);
    cudaDeviceSynchronize();

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString( err ));

    cudaMemset(d_lineCounter, 0, sizeof(int));
    findLinesKernel<<<findLinesGridDim, findLinesBlockDim>>>(nRows, nCols,
        d_accumulator, d_lines, d_lineCounter);
    cudaDeviceSynchronize();

    cudaMemcpy(&lineCounter, d_lineCounter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(lines, d_lines, 2 * MAX_NUM_LINES * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < lineCounter - 1; i += 2) {
        outputLines.push_back(Line(lines[i], lines[i + 1]));
    }
}


HoughTransformDevice::HoughTransformDevice(int frameWidth, int frameHeight) {
    nRows = (int) ceil(sqrt(frameHeight * frameHeight + frameWidth * frameWidth)) * 2 / RHO_STEP_SIZE;
    nCols = (THETA_B -THETA_A + (2*THETA_VARIATION)) / THETA_STEP_SIZE;

    frameSize = frameWidth * frameHeight * sizeof(uchar);
    cudaMallocHost(&(lines), 2 * MAX_NUM_LINES * sizeof(int));
    lineCounter = 0;

    cudaMalloc(&d_lines, 2 * MAX_NUM_LINES * sizeof(int));
    cudaMalloc(&d_lineCounter, sizeof(int));
    cudaMalloc(&d_frame, frameSize);
    cudaMalloc(&d_accumulator, nRows * nCols * sizeof(int));

    houghBlockDim = dim3(32, 5, 5);
    houghGridDim = dim3(ceil(frameHeight / 5), ceil(frameWidth / 5));
    findLinesBlockDim = dim3(32, 32);
    findLinesGridDim = dim3(ceil(nRows / 32), ceil(nCols / 32));
}


HoughTransformDevice::~HoughTransformDevice() {
    cudaFree(d_lines);
    cudaFree(d_lineCounter);
    cudaFree(d_frame);
    cudaFree(d_accumulator);
    cudaFreeHost(lines);
}
