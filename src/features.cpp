#include "features.hpp"

SIFT::Features::Features(std::vector<cv::KeyPoint> &key_points,
                         cv::Mat &descriptors,
                         std::vector<cv::DMatch> &good_matches)
    : key_points{key_points}, descriptors{descriptors}, good_matches{
                                                            good_matches} {}

void SIFT::Features::detectKeyPoints(const cv::Mat &img) {
  // auto detector = cv::SiftFeatureDetector::create();
  std::vector<cv::KeyPoint> kpts;
  detector->detect(img, kpts);
  key_points = kpts;
}

void SIFT::Features::extractDescriptors(
    const cv::Mat &img, const std::vector<cv::KeyPoint> &key_points) {
  // auto feature_extractor = cv::SiftDescriptorExtractor::create();
  cv::Mat des;
  auto kpts = key_points;
  feature_extractor->compute(img, kpts, des);
  descriptors = des;
}

void SIFT::Features::detectAndExtract(const cv::Mat &img) {
  // auto detectorAndExtractor = cv::SIFT::create();
  std::vector<cv::KeyPoint> kpts;
  cv::Mat des;
  detectorAndExtractor->detectAndCompute(img, cv::noArray(), kpts, des);
  key_points = kpts;
  descriptors = des;
}

void SIFT::Features::matchFeatures(const cv::Mat &descriptor1,
                                   const cv::Mat &descriptor2) {
  // auto matcher =
  //     cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  std::vector<std::vector<cv::DMatch>> knn_matches;
  std::vector<cv::DMatch> matches;
  matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);

  const float ratio_thresh = 0.75f;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance <
        ratio_thresh * knn_matches[i][1].distance) {
      matches.push_back(knn_matches[i][0]);
    }
  }
  good_matches = matches;
}

void SIFT::Features::findCorrespondences(const cv::Mat &img1,
                                         const cv::Mat &img2, bool show) {
  detectAndExtract(img1);
  auto kpts1 = getKeyPoints();
  auto descriptor1 = getDescriptors();

  detectAndExtract(img2);
  auto kpts2 = getKeyPoints();
  auto descriptor2 = getDescriptors();

  matchFeatures(descriptor1, descriptor2);
  auto matches = getMatches();

  if (show) {
    cv::Mat img_matches;
    cv::drawMatches(img1, kpts1, img2, kpts2, matches, img_matches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //-- Show detected matches
    cv::imshow("Good Matches", img_matches);
    cv::waitKey();
  }
  cv::destroyAllWindows();
}