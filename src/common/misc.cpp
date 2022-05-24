#include "misc.h"

KEY resolvedKey(const int key)
{
  switch (key) {
    case KEY_ESC:
    case KEY_SPACE:
    case KEY_ENTER:
    case KEY_DOWN_ARROW:
    case KEY_RIGHT_ARROW:
    case KEY_UP_ARROW:
    case KEY_LEFT_ARROW:
      return static_cast<KEY>(key);
    default:
      return KEY::NONE;
  }
}

cv::Mat scaleImage(const cv::Mat &image, const double scale, const int type)
{
  const cv::Size scaledSize(image.cols * scale, image.rows * scale);
  cv::Mat scaled(scaledSize, type, cv::Scalar::all(0));
  cv::resize(image, scaled, scaledSize, scale, scale, cv::INTER_CUBIC);
  return scaled;
}

cv::Mat newGray(const cv::Size &size, const cv::Scalar &color)
{
  cv::Mat gray(size, CV_8UC1, color);
  return gray;
}

cv::Mat newColor(const cv::Size &size, const cv::Scalar &color)
{
  cv::Mat colorImage(size, CV_8UC3, color);
  return colorImage;
}

std::vector<cv::Mat> getVideoFrames(cv::VideoCapture &capture, const int scaleTarget = 600)
{
  std::vector<cv::Mat> frames;
  cv::Mat frame;

  capture >> frame;
  while (!frame.empty()) {
    const double scaleFactor = frame.cols > scaleTarget ? ((double) scaleTarget) / frame.cols : 1.0;
    cv::Mat resized = scaleImage(frame, scaleFactor);
    frames.push_back(frame.clone());

    capture >> frame;
  }

  return frames;
}

std::vector<cv::Mat> convertAll(const std::vector<cv::Mat> &frames, const cv::ColorConversionCodes conversionType = cv::COLOR_BGR2GRAY)
{
  std::vector<cv::Mat> convertedFrames;
  for (const auto &frame : frames) {
    cv::Mat converted;
    cv::cvtColor(frame, converted, conversionType);
    convertedFrames.push_back(converted);
  }
  return convertedFrames;
}