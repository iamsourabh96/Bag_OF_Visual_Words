#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>

namespace SIFT {

class Features {
private:
  std::vector<cv::KeyPoint> key_points;
  cv::Mat descriptors;
  std::vector<cv::DMatch> good_matches;

  struct cv::Ptr<cv::SIFT> detectorAndExtractor = cv::SIFT::create();
  struct cv::Ptr<cv::SIFT> detector = cv::SiftFeatureDetector::create();
  struct cv::Ptr<cv::SIFT> feature_extractor =
      cv::SiftDescriptorExtractor::create();

  struct cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

public:
  Features() = default;
  Features(std::vector<cv::KeyPoint> &key_points, cv::Mat &descriptors,
           std::vector<cv::DMatch> &good_matches);

  void detectKeyPoints(const cv::Mat &img);

  void extractDescriptors(const cv::Mat &img,

                          const std::vector<cv::KeyPoint> &key_points);
  void detectAndExtract(const cv::Mat &img);

  void matchFeatures(const cv::Mat &descriptor1, const cv::Mat &descriptor2);

  void findCorrespondences(const cv::Mat &img1, const cv::Mat &img2,
                           const bool show = false);

  auto getKeyPoints() const { return key_points; };
  auto getDescriptors() const { return descriptors; };
  auto getMatches() const { return good_matches; };
};
} // namespace SIFT