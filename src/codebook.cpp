#include "codebook.hpp"

CodeBook::CodeBook(const cv::Mat &codebook,
                   const std::filesystem::path &data_path,
                   const std::filesystem::path &binary_path)
    : codebook{codebook}, data_path{data_path}, binary_path{binary_path} {
  if (binary_path == "") {
    auto path = data_path;
    path /= "bin";
    this->binary_path = path;
  }
  Mat::Serialization serialization(data_path, binary_path);
  this->serialization = serialization;

      validPath_();
}

CodeBook::CodeBook(const std::filesystem::path &data_path,
                   const std::filesystem::path &binary_path)
    : CodeBook({}, data_path, binary_path) {}

int CodeBook::validPath_() {
  // TODO: Use Exception Handling instead
  if (data_path == "" || std::filesystem::exists(data_path) == 0)
    std::cout << "Error: Invalid Path" << std::endl;
  else if (binary_path == "" || std::filesystem::exists(binary_path) == 0)
    std::cout << "Error: Invalid Path" << std::endl;
  else
    valid_path = 1;
  return valid_path;
}

void CodeBook::loadFeatureBook_(const std::filesystem::path &image_ext,
                                const std::string &suffix) {
  feature_vectors = serialization.deserializeAll(image_ext, suffix);
  featurebook = feature_vectors.at(0);
  for (size_t i{1}; i < feature_vectors.size(); i++) {
    cv::vconcat(featurebook, feature_vectors.at(i), featurebook);
  }
}

void CodeBook::generate(const std::filesystem::path &image_ext,
                        const std::string &suffix) {
  if (num_words == 0) {
    std::cout << "ERROR: Set number of words for the codebook first using -> "
                 "setNumWords()"
              << std::endl;
    return;
  }

  loadFeatureBook_(image_ext, suffix);
  // Run kmeans to get codebook
  cv::kmeans(featurebook, num_words, labels,
             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                              10, 1.0),
             3, cv::KMEANS_PP_CENTERS, codebook);
}

void CodeBook::load(const std::filesystem::path &name) {
  codebook = serialization.deserialize(name);
}

void CodeBook::save(const std::filesystem::path &name) {
  serialization.serialize(codebook, name);
}

