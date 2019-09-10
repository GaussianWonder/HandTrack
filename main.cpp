#include <iostream>
#include <opencv2/highgui.hpp>

#include "./source/skinDetector.cpp"

#define CAMERA "Camera"
#define OUTPUT "Computer Vision"
#define CTHRESH "Combined Threshold"
#define EKERNEL "Erode Kernel"

using namespace std;
using namespace cv;

Mat frame, processed;
skinDetector detector;

static void getHsvThreshTrack(int pos, void *){
    detector.addHsvMask = (pos + 1);
}

static void getErodeKernTrack(int pos, void *){
    detector.erosionKernel = (pos + 1);
}

int main(int argc, char * argv[])
{
    /*Mat rand(100, 100, CV_8UC3, Scalar(0, 0, 1));
    
    for(int i=0; i<rand.rows; ++i){
        Vec3b *p = rand.ptr<Vec3b>(i);
        
        for(int j=0; j<rand.cols; ++j){
            Vec3b newP = Vec3b(2, 2, 2);
            p[j] = newP;
        }
        cout << '\n';
    }
    
    threshold(rand, rand, 0, 255, THRESH_BINARY);
    imshow(OUTPUT, rand);
    
    while(1)
        if(waitKey(10) == 27)
            break;*/
    namedWindow(CAMERA);
    namedWindow(OUTPUT);
    
    createTrackbar(CTHRESH, CAMERA, &detector.addHsvMask, 1000, getHsvThreshTrack);
    createTrackbar(EKERNEL, CAMERA, &detector.erosionKernel, 10, getHsvThreshTrack);
    
    VideoCapture cap(0);
    if(!cap.isOpened()){
        cout << "Failed to open the camera!\n";
        return -1;
    }
    
    while(1){
        //Capture next frame
        cap >> frame;
        if(frame.empty())
            continue;
        
        //Original
        imshow(CAMERA, frame);
        
        //Detect skin color
        detector(frame, processed);
        
        //Shou output after processing chain
        imshow(OUTPUT, processed);
        
        //Keyboard events
        int keyboard = waitKey(10);
        if(keyboard == 27)  //ESC (EXIT)
            break;
    }
    
    destroyAllWindows();
    return 0;
}
