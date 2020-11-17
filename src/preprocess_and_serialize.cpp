#include "codebook.hpp"
#include "histbook.hpp"
#include "serialization.hpp"

#include <filesystem>
#include <iostream>
#include <map>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

int main() {
  // Path from root project folder
  const fs::path data_path = fs::canonical("../data/kitti_Seq/Kitti_Seq_07");
  // const fs::path data_path = fs::canonical("../data/bunny_data/images");
  const int num_words = 1500;
  const std::string image_ext = ".png";
  // const std::string image_ext = ".ppm";
  const std::string suffix = "";

  Mat::Serialization serialization(data_path);
  SIFT::Features sift;

  // Read all images in data folder
  auto image_path = data_path;
  (image_path /= "*") += image_ext;
  std::vector<cv::String> imnames;
  cv::glob(image_path, imnames, false);

  for (auto &imname : imnames) {
    const cv::Mat image = cv::imread(imname, cv::IMREAD_COLOR);
    sift.detectAndExtract(image);
    cv::Mat descriptor = sift.getDescriptors();

    std::filesystem::path name = imname;
    name = (name.stem()) += suffix;
    serialization.serialize(descriptor, name); // Store bin to disk
  }

  CodeBook codebook(data_path);

  // Generate new code book
  codebook.setNumWords(num_words);
  codebook.generate(image_ext, suffix);

  // Save codebook to bin_path
  codebook.save("codebook");

  // Load pre-computed codebook
  // codebook.load("codebook");

  cv::Mat mycodebook = codebook.get();

  HistBook histbook(mycodebook, data_path);
  histbook.generate(image_ext,
                    suffix); // Compute histogram for all images in the dataset

  histbook.save("histbook");

  // Load saved histbook
  // std::map<std::string, std::vector<float>> loaded_histbook =
  //     histbook.load("histbook");
}