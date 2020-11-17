#include "serialization.hpp"
// #include <boost/filesystem.hpp>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

Mat::Serialization::Serialization(const std::filesystem::path &data_path,
                                  const std::filesystem::path &bin_path)
    : data_path{data_path}, binary_path{bin_path} {
  if (bin_path == "") {
    auto path = data_path;
    path /= "bin";
    if (!std::filesystem::is_directory(path)) {
      std::cout << "Bin Folder does not exist"
                << "\n"
                << "Creating new Bin Folder" << std::endl;
      std::filesystem::create_directory(path);
    }
    this->binary_path = path;
  }
  validPath_();
}

int Mat::Serialization::validPath_() {
  // TODO: Use Exception Handling instead
  if (data_path == "" || std::filesystem::exists(data_path) == 0)
    std::cout << "Error: Invalid Path" << std::endl;
  else if (binary_path == "" || std::filesystem::exists(binary_path) == 0)
    std::cout << "Error: Invalid Path" << std::endl;
  else
    valid_path = 1;
  return valid_path;
}

void Mat::Serialization::serialize(const cv::Mat &m,
                                   const std::filesystem::path &name) {
  auto bin_name = name;
  auto path = binary_path;
  if (!bin_name.has_extension())
    bin_name += ".bin";
  else if (bin_name.extension() != ".bin")
    bin_name = (bin_name.stem()) += ".bin";
  path /= bin_name;

  std::ofstream file(path.c_str(), std::ios::binary);
  cereal::BinaryOutputArchive ar(file);
  ar(m);
}

void Mat::Serialization::serialize(const std::filesystem::path &name,
                                   const std::string &suffix) {
  // if ext is provided - read all files with given extension
  if (((std::string)name)[0] == '.') {
    std::vector<cv::String> imnames;
    auto path = data_path;
    (path /= "*") += name;
    cv::glob(path, imnames, false);

    for (int i{0}; i < imnames.size(); i++) {
      std::filesystem::path filename = imnames.at(i);
      const cv::Mat image = cv::imread(imnames.at(i), cv::IMREAD_COLOR);

      auto bin_name = ((filename.stem()) += suffix) += ".bin";
      serialize(image, bin_name);
    }
  } else {
    cv::Mat image = cv::imread(name);
    auto bin_name = ((name.stem()) += suffix) += ".bin";
    serialize(image, bin_name);
  }
}

cv::Mat Mat::Serialization::deserialize(const std::filesystem::path &name) {
  cv::Mat loaded_data;
  auto bin_name = name;
  if (!bin_name.has_extension())
    bin_name += ".bin";
  else if (bin_name.extension() != ".bin")
    bin_name = (bin_name.stem()) += ".bin";

  auto path = binary_path;
  path /= bin_name;

  std::ifstream file(path.c_str(), std::ios::binary);
  cereal::BinaryInputArchive ar(file);
  ar(loaded_data);
  return loaded_data;
}

std::vector<cv::Mat>
Mat::Serialization::deserializeAll(const std::filesystem::path &ext,
                                   const std::string &suffix) {
  std::vector<cv::Mat> loaded_data;
  // Redundant if statement
  if (ext == "") {
    for (auto &file : std::filesystem::directory_iterator(binary_path))
      loaded_data.emplace_back(deserialize(file));

  } else if (((std::string)ext)[0] == '.') {
    std::vector<cv::String> imnames;
    auto path = data_path;
    (path /= "*") += ext;
    cv::glob(path, imnames, false);

    for (int i{0}; i < imnames.size(); i++) {
      std::filesystem::path filename = imnames.at(i);
      auto bin_path = (filename.stem()) += suffix;
      loaded_data.push_back(deserialize(bin_path));
    }
  } else {
    std::cout << "Enter valid extension stating with .";
  }
  return loaded_data;
}

void Mat::Serialization::setPath(const std::filesystem::path &data_path,
                                 const std::filesystem::path &binary_path) {
  this->data_path = data_path;
  if (binary_path == "") {
    auto path = data_path;
    path /= "bin";
    if (!std::filesystem::is_directory(path)) {
      std::cout << "Bin Folder does not exist"
                << "\n"
                << "Creating new Bin Folder" << std::endl;
      std::filesystem::create_directory(path);
    }
    this->binary_path = path;
  }
}
