#include "misc.h"
#include "logger.h"
#include <algorithm>

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

std::vector<cv::Mat> getVideoFrames(cv::VideoCapture &capture, const int widthTarget, const int heightTarget)
{
  std::vector<cv::Mat> frames;
  cv::Mat frame;

  capture >> frame;
  while (!frame.empty()) {
    const double widthScaleFactor = frame.cols > widthTarget ? ((double) widthTarget) / frame.cols : 1.0;
    const double heightScaleFactor = frame.rows > heightTarget ? ((double) heightTarget) / frame.rows : 1.0;
    const double scaleFactor = MIN(widthScaleFactor, heightScaleFactor);

    cv::Mat resized = scaleImage(frame, scaleFactor);
    frames.push_back(resized.clone());

    capture >> frame;
  }

  return frames;
}

std::vector<cv::Mat> convertAll(const std::vector<cv::Mat> &frames, const cv::ColorConversionCodes conversionType)
{
  std::vector<cv::Mat> convertedFrames;
  for (const auto &frame : frames) {
    cv::Mat converted;
    cv::cvtColor(frame, converted, conversionType);
    convertedFrames.push_back(converted);
  }
  return convertedFrames;
}

ObjectTrace::ObjectTrace(const cv::Mat &binaryImage)
{
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  std::size_t largestContour = 0; // largest contour === largest area for this usecase
  double maxArea = -1;
  for (std::size_t i = 0; i < contours.size(); ++i) {
    const double area = cv::contourArea(contours[i]);
    if (area > maxArea) {
      maxArea = area;
      largestContour = i;
    }
  }

  INFO("index largest {} with area {}, total contours: {}", largestContour, maxArea, contours.size());

  if (maxArea > 0) {
    std::copy(contours[largestContour].begin(), contours[largestContour].end(), std::back_inserter(this->contour));
    cv::Mat contourMat(contours[largestContour]);

    std::vector<cv::Point> hull;
    cv::convexHull(contourMat, hull, true);
    std::copy(hull.begin(), hull.end(), std::back_inserter(this->hull));
    this->area = cv::contourArea(hull);

    std::vector<int> hullIndexes;
    cv::convexHull(contourMat, hullIndexes, true);

    std::vector<cv::Vec4i> convexityDefects;
    cv::convexityDefects(contourMat, hullIndexes, convexityDefects);
    std::copy(convexityDefects.begin(), convexityDefects.end(), std::back_inserter(this->hullDefects));
  }
}

cv::Mat ObjectTrace::draw(const cv::Size &size)
{
  cv::Mat drawing = newColor(size);
  if (this->area <= 0)
    return drawing;

  cv::RNG RNG((int) time(0));

  std::vector<std::vector<cv::Point>> contours;
  std::vector<std::vector<cv::Point>> hulls;

  contours.emplace_back(std::vector<cv::Point>(this->contour));
  hulls.push_back(std::vector<cv::Point>(this->hull));

  cv::drawContours(drawing, contours, 0, cv::Scalar(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256)));
  cv::drawContours(drawing, hulls, 0, cv::Scalar(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256)));

  cv::Scalar defectColor(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256));
  cv::Scalar defectLine(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256));
  for(auto &defect : this->hullDefects) {
    cv::Point ptStart(this->contour[defect[0]]);
    cv::Point ptEnd(this->contour[defect[1]]);
    cv::Point ptFar(this->contour[defect[2]]);

    cv::line(drawing, ptStart, ptEnd, defectLine);
    cv::circle(drawing, ptFar, 5, defectColor);
  }

  return drawing;
}
