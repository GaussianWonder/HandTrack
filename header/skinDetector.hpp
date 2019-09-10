#pragma once

#include <iostream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

#include <opencv2/highgui.hpp>

class skinDetector{
public:
    void operator()(Mat &src, Mat &dst){
        medianBlur(src, dst, 3);
        
        //Get masks
        Mat hsvM(hsvMask(dst)),
            rgbM(rgbMask(dst)),
            ycrbrM(ycrcbMask(dst)),
            temp;
                    
        //Average masks and apply last threshold
        dst = Mat(src.size(), CV_8UC1, Scalar(0));
        
        dst += (hsvM / (int(1000 / this->addHsvMask) + 1) ) + rgbM + ycrbrM;
        dst /= 3;
        
        threshold(dst, dst, 0, 255, THRESH_BINARY);

        dst = closing(dst);
        dst = cutMask(src, dst);
    }

    int addHsvMask = 1000;
    int erosionKernel = 3;
private:
    Mat hsvMask(Mat &img, unsigned char thresh = 127);
    Mat rgbMask(Mat &img);
    Mat ycrcbMask(Mat &img, unsigned char thresh = 127);
    Mat cutMask(Mat &img, Mat &mask);
    Mat closing(Mat &mask);
    
};
