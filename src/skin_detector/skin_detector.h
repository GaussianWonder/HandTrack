#ifndef __SKIN_DETECTOR_H__
#define __SKIN_DETECTOR_H__

#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>

class SkinDetector
{
public:
  void operator()(cv::Mat &src, cv::Mat &dst)
  {
    cv::medianBlur(src, dst, 3);

    // Get masks
    cv::Mat hsvM(hsvMask(dst)),
        rgbM(rgbMask(dst)),
        ycrbrM(ycrcbMask(dst)),
        temp;

    // Average masks and apply last threshold
    dst = cv::Mat(src.size(), CV_8UC1, cv::Scalar(0));

    cv::imshow("HSV", hsvM);
    cv::imshow("RGB", rgbM);
    cv::imshow("YCRBR", ycrbrM);
    dst += (hsvM / (int(1000 / this->addHsvMask) + 1)) + rgbM + ycrbrM;
    dst /= 3;

    threshold(dst, dst, 0, 255, cv::THRESH_BINARY);

    dst = closing(dst);
    dst = cutMask(src, dst);
  }

  int addHsvMask = 1000;
  int erosionKernel = 3;

private:
  cv::Mat hsvMask(cv::Mat &img, unsigned char thresh = 127);
  cv::Mat rgbMask(cv::Mat &img);
  cv::Mat ycrcbMask(cv::Mat &img, unsigned char thresh = 127);
  cv::Mat cutMask(cv::Mat &img, cv::Mat &mask);
  cv::Mat closing(cv::Mat &mask);
};

#endif // __SKIN_DETECTOR_H__
