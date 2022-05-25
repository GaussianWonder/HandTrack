#ifndef __MISC_H__
#define __MISC_H__

#include "opencv2/opencv.hpp"
#include <string>

// Key definitions
#define KEY_ESC 27
#define KEY_SPACE 32
#define KEY_ENTER 13
// Maybe KEY_RETURN for MAC?

#define KEY_DOWN_ARROW 84
#define KEY_RIGHT_ARROW 83
#define KEY_UP_ARROW 82
#define KEY_LEFT_ARROW 81

// Key enum (prefer to use this)
enum KEY {
  ESC = KEY_ESC,
  SPACE = KEY_SPACE,
  ENTER = KEY_ENTER,

  DOWN_ARROW = KEY_DOWN_ARROW,
  RIGHT_ARROW = KEY_RIGHT_ARROW,
  UP_ARROW = KEY_UP_ARROW,
  LEFT_ARROW = KEY_LEFT_ARROW,

  NONE = 0,
};

KEY resolvedKey(const int key);

// Wait for key press and return a consistent key across platforms
#define WaitKey(D) (cv::waitKey(D) & 0xEFFFFF)

// Reading And Converting Images/Videos
std::vector<cv::Mat> getVideoFrames(cv::VideoCapture &capture, const int widthTarget = 600, const int heightTarget = 600);
std::vector<cv::Mat> convertAll(const std::vector<cv::Mat> &frames, const cv::ColorConversionCodes conversionType = cv::COLOR_BGR2GRAY);

// Image scaling

cv::Mat scaleImage(const cv::Mat &image, const double scale, const int type = CV_8UC3);
cv::Mat newGray(const cv::Size &size, const cv::Scalar &color = cv::Scalar::all(0));
cv::Mat newColor(const cv::Size &size, const cv::Scalar &color = cv::Scalar::all(0));

/**
 * @brief The trace of an object
 * This is drawable too
 */
class ObjectTrace {
public:
  std::vector<cv::Point> contour;
  std::vector<cv::Point> hull;
  std::vector<cv::Vec4i> hullDefects;

  cv::Point hullCenter;
  cv::Point contourCenter;

  std::vector<cv::Vec4i> hierarchy;

  double area = -1;

  ObjectTrace(){}
  ObjectTrace(const cv::Mat &binaryImage);

  cv::Mat draw(const cv::Size &size);
};

float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1);
float innerAngle(const cv::Point &p1, const cv::Point &p2, const cv::Point &p3);

cv::Point midpoint(const cv::Point& a, const cv::Point& b);

#endif