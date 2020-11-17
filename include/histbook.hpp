#pragma once

#include "codebook.hpp"
#include <map>

class HistBook {
private:
  cv::Mat codebook;
  std::filesystem::path data_path, binary_path;

  int histogram_length;
  int histbook_size;

  std::vector<int> word_occurances;
  std::map<std::string, std::vector<int>> histbook_raw;
  std::map<std::string, std::vector<double>> histbook;

  SIFT::Features sift;
  std::vector<cv::DMatch> matches;

  Mat::Serialization deserialize;

  int valid_path = 0;
  int isvalidPath_();
  std::vector<int> computeHist_(const cv::Mat &des);
  // Computes histogram for all images with provided ext before TF-IDF
  void computeHistAll_(const std::filesystem::path &ext,
                       const std::string &suffix = "");

  std::vector<double> TF_IDF_(const std::vector<int> &hist);

  std::vector<std::string> KNMatcher_(const std::vector<double> &query_hist,
                                      const int k);

public:
  HistBook(const cv::Mat &codebook, const std::filesystem::path &data_path,
           const std::filesystem::path &binary_path = "");

  explicit HistBook(const std::filesystem::path &data_path,
                    const std::filesystem::path &binary_path = "");

  void loadCodeBook(const std::filesystem::path &name) {
    codebook = deserialize.deserialize(name);
  }

  // Computes histogram of the image provided, with ref to the codebook
  std::vector<int> computeHist(const cv::Mat &image);
  std::vector<int> computeHist(const std::filesystem::path &name);
  void generate();

  // Displays histogram on terminal window
  void displayHist(const std::vector<int> &hist);
  void displayHist(const std::vector<double> &hist);
  void displayHist(const cv::Mat &image);
  void displayHist(const std::filesystem::path &name);

  void generate(const std::filesystem::path &image_ext,
                const std::string &suffix = "");

  void save(const std::filesystem::path &name, const std::string &suffix = "");

  std::map<std::string, std::vector<double>>
  load(const std::filesystem::path &name, const std::string &suffix = "");

  std::vector<std::string> KNMatcher(const cv::Mat &query_image, const int &k);
  std::vector<std::string> KNMatcher(const std::filesystem::path &filename,
                                     const int &k);

  std::map<std::string, std::vector<double>> getHistBook() const {
    return histbook;
  };
  std::map<std::string, std::vector<int>> getHistBookRaw() const {
    return histbook_raw;
  };
};
