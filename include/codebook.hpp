#pragma once

#include "serialization.hpp"

class CodeBook {
private:
  cv::Mat codebook, featurebook, labels;
  std::vector<cv::Mat> feature_vectors;
  int num_words{0};

  std::filesystem::path data_path = "";
  std::filesystem::path binary_path = "";

  Mat::Serialization serialization;

  int valid_path = 0;
  int validPath_();
  // Assuming features are already computed and stored for all images in the
  // data folder. Given the extension of images in the data folder - will
  // deserialize all the computed features for all images with provided
  // extension. Stack them together into one Mat and store in featurebook.
  void loadFeatureBook_(const std::filesystem::path &image_ext,
                        const std::string &suffix = "");

public:
  CodeBook(const cv::Mat &codebook, const std::filesystem::path &data_path,
           const std::filesystem::path &bin_path = "");

  explicit CodeBook(const std::filesystem::path &data_path,
                    const std::filesystem::path &bin_path = "");

  // Sets the number of words to generate codebook
  void setNumWords(const int &num_words) { this->num_words = num_words; };

  // Generates a new codebook including all images with provided ext
  void generate(const std::filesystem::path &image_ext,
                const std::string &suffix = "");

  // Loads codebook from binary_path
  void load(const std::filesystem::path &name);

  // Saves codebook to binary path
  void save(const std::filesystem::path &name);

  // Returns codebook generated for the current instance
  cv::Mat get() const { return codebook; };
  std::vector<cv::Mat> getFeatureVectors() const { return feature_vectors; };
};