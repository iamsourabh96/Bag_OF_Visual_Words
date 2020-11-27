#include "codebook.hpp"
#include "histbook.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

int main() {
  const fs::path data_path =
      fs::canonical("/home/sourabh/stuff/projects/Bag_Of_Visual_Words/data/"
                    "kitti_Seq/Kitti_Seq_07");
  // const fs::path data_path = fs::canonical("../data/bunny_data/images");
  // const fs::path image_name = "0023.ppm";
  const fs::path image_name = "000000.png";
  int k = 20; // Num of closest match to search for.

  std::filesystem::path image_path = data_path;
  image_path /= image_name;

  cv::Mat query_image = cv::imread(image_path, cv::IMREAD_COLOR);

  CodeBook codebook{data_path};
  codebook.load("codebook"); // Load pre-computed codebook
  cv::Mat mycodebook = codebook.get();

  HistBook histbook(mycodebook, data_path);
  histbook.load("histbook"); // Load saved histbook

  std::vector<std::string> kmatches = histbook.KNMatcher(query_image, k);

  for (const auto &match : kmatches) {
    std::cout << match << std::endl;
  }
}
