#include "skin_detector.h"

cv::Mat SkinDetector::hsvMask(cv::Mat &img, unsigned char thresh)
{
  // Lower threshold 0    50  0
  // Upper threshold 120  255 255
  cv::Mat mask;

  cv::cvtColor(img, mask, cv::COLOR_RGB2HSV);
  cv::inRange(mask, cv::Scalar(0, 50, 0), cv::Scalar(120, 255, 255), mask);
  cv::threshold(mask, mask, thresh, 255, cv::THRESH_BINARY);

  return mask;
}

cv::Mat SkinDetector::rgbMask(cv::Mat &img)
{
  cv::Mat mask(img.size(), CV_8UC3);
  cv::Vec3b on = cv::Vec3b(255, 255, 255),
            off = cv::Vec3b(0, 0, 0);

  for (int i = 0; i < img.rows; ++i)
  {
    cv::Vec3b *imgP = img.ptr<cv::Vec3b>(i);
    cv::Vec3b *maskP = mask.ptr<cv::Vec3b>(i);

    for (int j = 0; j < img.cols; ++j)
      if ((imgP[j][2] > 95 && imgP[j][1] > 40 && imgP[j][0] > 20 &&
           (MAX(imgP[j][0], MAX(imgP[j][1], imgP[j][2])) - MIN(imgP[j][0], MIN(imgP[j][1], imgP[j][2])) > 15) &&
           abs(imgP[j][2] - imgP[j][1]) > 15 && imgP[j][2] > imgP[j][1] && imgP[j][1] > imgP[j][0]) ||
          (imgP[j][2] > 200 && imgP[j][1] > 210 && imgP[j][0] > 170 && abs(imgP[j][2] - imgP[j][1]) <= 15 &&
           imgP[j][2] > imgP[j][0] && imgP[j][1] > imgP[j][0]))
        maskP[j] = on;
      else
        maskP[j] = off;
  }

  cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
  return mask;
}

cv::Mat SkinDetector::ycrcbMask(cv::Mat &img, unsigned char thresh)
{
  // Lower threshold 0    133 77
  // Upper threshold 235   173 127
  cv::Mat mask;

  cv::cvtColor(img, mask, cv::COLOR_BGR2YCrCb);
  cv::inRange(mask, cv::Scalar(0, 133, 77), cv::Scalar(235, 173, 127), mask);
  cv::threshold(mask, mask, thresh, 255, cv::THRESH_BINARY);

  return mask;
}

cv::Mat SkinDetector ::cutMask(cv::Mat &img, cv::Mat &mask)
{
  cv::Mat dst, free;
  cv::filter2D(
      mask, dst, -1,
      cv::Mat::ones(70, 70, CV_32F) / (float)(70 * 70));

  // Make everything white except pure black
  cv::bitwise_not(dst, free);

  cv::Mat grabMask(img.size(), CV_8UC1, cv::Scalar(2));

  for (int i = 0; i < img.rows; ++i)
  {
    uchar *maskP = mask.ptr<uchar>(i);
    uchar *freeP = free.ptr<uchar>(i);
    uchar *grabMaskP = grabMask.ptr<uchar>(i);

    for (int j = 0; j < img.cols; ++j)
    {
      if (int(maskP[j]) == 255)
        grabMaskP[j] = 255;

      if (int(freeP[j]) == 255 || int(grabMaskP[j]) == 2)
        grabMaskP[j] = 0;
    }
  }

  return grabMask;
}

cv::Mat SkinDetector ::closing(cv::Mat &mask)
{
  cv::morphologyEx(
    mask, mask,
    cv::MORPH_CLOSE,
    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5))
  );

  cv::morphologyEx(
    mask, mask,
    cv::MORPH_OPEN,
    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)),
    cv::Point(-1, -1),
    2
  );

  cv::Mat kernel = cv::getStructuringElement(
    cv::MORPH_ELLIPSE,
    cv::Size(2 * this->erosionKernel + 1, 2 * this->erosionKernel + 1),
    cv::Point(this->erosionKernel, this->erosionKernel)
  );

  cv::erode(mask, mask, kernel);

  return mask;
}
