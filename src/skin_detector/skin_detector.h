#ifndef __SKIN_DETECTOR_H__
#define __SKIN_DETECTOR_H__

#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>

class SkinDetector
{
public:
  void operator()(const cv::Mat &src, cv::Mat &dst)
  {
    cv::medianBlur(src, dst, 3);

    // Get masks
    cv::Mat hsvM(hsvMask(dst)),
        rgbM(rgbMask(dst)),
        ycrbrM(ycrcbMask(dst));

    // Average masks and apply last threshold
    dst = cv::Mat(src.size(), CV_8UC1, cv::Scalar(0));

    // cv::imshow("HSV", hsvM);
    // cv::imshow("RGB", rgbM);
    // cv::imshow("YCRBR", ycrbrM);

    cv::bitwise_or(rgbM, ycrbrM, dst);

    cv::threshold(dst, dst, 127, 255, cv::THRESH_BINARY);

    // cv::imshow("RGB | YCRBR", dst);

    dst = closing(dst).clone();
    dst = cutMask(src, dst).clone();
  }

  int openCloseKernel = 6;
  int blurKernel = 2;
  int erodeKernel = 2;

private:
  cv::Mat hsvMask(const cv::Mat &img, unsigned char thresh = 127);
  cv::Mat rgbMask(const cv::Mat &img);
  cv::Mat ycrcbMask(const cv::Mat &img, unsigned char thresh = 127);
  cv::Mat cutMask(const cv::Mat &img, cv::Mat &mask);
  cv::Mat closing(const cv::Mat &mask);
};

#endif // __SKIN_DETECTOR_H__
