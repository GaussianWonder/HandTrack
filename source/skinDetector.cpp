#include "../header/skinDetector.hpp"

using namespace std;
using namespace cv;

Mat skinDetector :: hsvMask(Mat &img, unsigned char thresh){
    //Lower threshold 0    50  0
    //Upper threshold 120  255 255
    Mat mask;
    
    cvtColor(img, mask, COLOR_RGB2HSV);
    inRange(mask, Scalar(0, 50, 0), Scalar(120, 255, 255), mask);
    threshold(mask, mask, thresh, 255, THRESH_BINARY);
    
    return mask;
}

Mat skinDetector :: rgbMask(Mat &img){
    //++Weird ass condition
    Mat mask(img.size(), CV_8UC3);
    Vec3b on = Vec3b(255, 255, 255),
          off = Vec3b(0, 0, 0);
    
    for(int i=0; i<img.rows; ++i){
        Vec3b *imgP     = img.ptr<Vec3b>(i);
        Vec3b *maskP   = mask.ptr<Vec3b>(i);
        
        for(int j=0; j<img.cols; ++j)
            if ((imgP[j][2] > 95 && imgP[j][1]>40 && imgP[j][0] > 20 &&
                    (MAX(imgP[j][0], MAX(imgP[j][1], imgP[j][2])) - MIN(imgP[j][0], MIN(imgP[j][1], imgP[j][2])) > 15) &&
                    abs(imgP[j][2] - imgP[j][1]) > 15 && imgP[j][2] > imgP[j][1] && imgP[j][1] > imgP[j][0]) ||
                    (imgP[j][2] > 200 && imgP[j][1] > 210 && imgP[j][0] > 170 && abs(imgP[j][2] - imgP[j][1]) <= 15 &&
                    imgP[j][2] > imgP[j][0] &&  imgP[j][1] > imgP[j][0]))
                maskP[j] = on;
            else
                maskP[j] = off;
    }
    
    cvtColor(mask, mask, COLOR_BGR2GRAY);
    return mask;
}

Mat skinDetector :: ycrcbMask(Mat &img, unsigned char thresh){
    //Lower threshold 0    133 77
    //Upper threshold 235   173 127
    Mat mask;
    
    cvtColor(img, mask, COLOR_BGR2YCrCb);
    inRange(mask, Scalar(0, 133, 77), Scalar(235, 173, 127), mask);
    threshold(mask, mask, thresh, 255, THRESH_BINARY);
    
    return mask;
}

Mat skinDetector :: cutMask(Mat &img, Mat &mask){
    Mat dst, free;
    filter2D(
        mask, dst, -1,
        Mat::ones(70, 70, CV_32F) / (float)(70 * 70)
    );
    
    //Make everything white except pure black
    bitwise_not(dst, free);
    
    Mat grabMask(img.size(), CV_8UC1, Scalar(2));
    
    for(int i=0; i<img.rows; ++i){
        uchar *maskP     = mask.ptr<uchar>(i);
        uchar *freeP     = free.ptr<uchar>(i);
        uchar *grabMaskP = grabMask.ptr<uchar>(i);
        
        for(int j=0; j<img.cols; ++j){
            if(int(maskP[j]) == 255)
                grabMaskP[j] = 255;
            
            if(int(freeP[j]) == 255 || int(grabMaskP[j]) == 2)
                grabMaskP[j] = 0;
        }
    }
    
    return grabMask;
}

Mat skinDetector :: closing(Mat &mask){
    morphologyEx(
        mask, mask,
        MORPH_CLOSE,
        getStructuringElement(MORPH_ELLIPSE, Size(5, 5))
    );
    
    morphologyEx(
        mask, mask,
        MORPH_OPEN,
        getStructuringElement(MORPH_ELLIPSE, Size(3, 3)),
        Point(-1, -1),
        2
    );
    
    Mat kernel = getStructuringElement(
        MORPH_ELLIPSE,
        Size(2 * this->erosionKernel + 1, 2 * this->erosionKernel + 1),
        Point(this->erosionKernel, this->erosionKernel)
    );
    
    erode(mask, mask, kernel);
    
    return mask;
}


