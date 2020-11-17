#pragma once

#ifndef MATCEREALISATION_HPP_
#define MATCEREALISATION_HPP_

#include "opencv2/core/core.hpp"
#include <cereal/archives/binary.hpp>

#include <filesystem>
#include <string>

#include "features.hpp"

/**
 * Serialisation for OpenCV cv::Mat matrices for the serialisation
 * library cereal (http://uscilab.github.io/cereal/index.html).
 */

namespace cv {

/**
 * Serialise a cv::Mat using cereal.
 *
 * Supports all types of matrices as well as non-contiguous ones.
 *
 * @param[in] ar The archive to serialise to.
 * @param[in] mat The matrix to serialise.
 */
template <class Archive> void save(Archive &ar, const cv::Mat &mat) {
  int rows, cols, type;
  bool continuous;

  rows = mat.rows;
  cols = mat.cols;
  type = mat.type();
  continuous = mat.isContinuous();

  ar &rows &cols &type &continuous;

  if (continuous) {
    const int data_size = rows * cols * static_cast<int>(mat.elemSize());
    auto mat_data = cereal::binary_data(mat.ptr(), data_size);
    ar &mat_data;
  } else {
    const int row_size = cols * static_cast<int>(mat.elemSize());
    for (int i = 0; i < rows; i++) {
      auto row_data = cereal::binary_data(mat.ptr(i), row_size);
      ar &row_data;
    }
  }
};

/* De-serialise a cv::Mat using cereal.
 *
 * Supports all types of matrices as well as non-contiguous ones.
 *
 * @param[in] ar The archive to deserialise from.
 * @param[in] mat The matrix to deserialise into.
 */
template <class Archive> void load(Archive &ar, cv::Mat &mat) {
  int rows, cols, type;
  bool continuous;

  ar &rows &cols &type &continuous;

  if (continuous) {
    mat.create(rows, cols, type);
    const int data_size = rows * cols * static_cast<int>(mat.elemSize());
    auto mat_data = cereal::binary_data(mat.ptr(), data_size);
    ar &mat_data;
  } else {
    mat.create(rows, cols, type);
    const int row_size = cols * static_cast<int>(mat.elemSize());
    for (int i = 0; i < rows; i++) {
      auto row_data = cereal::binary_data(mat.ptr(i), row_size);
      ar &row_data;
    }
  }
};

} // namespace cv

namespace Mat {

class Serialization {
private:
  std::filesystem::path data_path = "";
  std::filesystem::path binary_path = "";
  int valid_path = 0;

  int validPath_();

public:
  Serialization() = default;
  explicit Serialization(const std::filesystem::path &data_path,
                         const std::filesystem::path &bin_path = "");
  Serialization(const Serialization &) = default; // Copy constructor

  // Serializes - provided Mat file to bin_path
  void serialize(const cv::Mat &m, const std::filesystem::path &name);

  // Reads a Mat image from data_path and then serializes to bin_path
  // if ext is provided - serializes all images with the extension
  void serialize(const std::filesystem::path &name,
                 const std::string &suffix = "");

  // Deserializes bin file to Mat
  // name can be full path or just the stem
  cv::Mat deserialize(const std::filesystem::path &name);

  // Deserializes all the bin files in the binary path if no name is provided.
  // If ext of the images is provided (eg: .jpg) will return the corresponding
  // bin file/files.
  std::vector<cv::Mat> deserializeAll(const std::filesystem::path &ext,
                                      const std::string &suffix = "");

  // If no args constructor is used
  void setPath(const std::filesystem::path &data_path,
               const std::filesystem::path &binary_path = "");
};

} // namespace Mat

#endif /* MATCEREALISATION_HPP_ */