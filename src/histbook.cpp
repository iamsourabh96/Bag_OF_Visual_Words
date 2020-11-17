#include "histbook.hpp"
#include <algorithm>
#include <fstream>
#include <numeric>

HistBook::HistBook(const cv::Mat &codebook,
                   const std::filesystem::path &data_path,
                   const std::filesystem::path &binary_path)
    : codebook{codebook}, data_path{data_path}, binary_path{binary_path} {

  if (binary_path == "") {
    auto path = data_path;
    path /= "bin";
    this->binary_path = path;
  }
  histogram_length = codebook.rows;
  std::vector<int> occurances(histogram_length, 0);
  word_occurances = occurances;

  isvalidPath_();

  Mat::Serialization deserialize(data_path, binary_path);
  this->deserialize = deserialize;
}

HistBook::HistBook(const std::filesystem::path &data_path,
                   const std::filesystem::path &bin_path)
    : HistBook({}, data_path, bin_path) {}

int HistBook::isvalidPath_() {
  // TODO: Use Exception Handling instead
  if (data_path == "" || std::filesystem::exists(data_path) == 0)
    std::cout << "Error: Invalid Path" << std::endl;
  else if (binary_path == "" || std::filesystem::exists(binary_path) == 0)
    std::cout << "Error: Invalid Path" << std::endl;
  else
    valid_path = 1;
  return valid_path;
}

std::vector<int> HistBook::computeHist_(const cv::Mat &des) {
  if (!codebook.rows)
    std::cout << "ERROR: CodeBook Loading Error" << std::endl;

  sift.matchFeatures(des, codebook);
  matches = sift.getMatches();

  std::vector<int> histogram(histogram_length, 0);
  for (const auto &match : matches) {
    int idx = match.trainIdx;
    histogram.at(idx) += 1;
  }
  return histogram;
}

void HistBook::computeHistAll_(const std::filesystem::path &ext,
                               const std::string &suffix) {
  if (((std::string)ext)[0] != '.') {
    std::cout << "Enter valid extension stating with .";
    return;
  }

  std::vector<cv::String> imnames;
  auto path = data_path;
  (path /= "*") += ext;
  cv::glob(path, imnames, false);

  for (int i{0}; i < imnames.size(); i++) {
    std::filesystem::path filename = imnames.at(i);
    std::string name = (filename.stem()) += suffix;
    cv::Mat descriptor = deserialize.deserialize(name);

    std::vector<int> histogram = computeHist_(descriptor);
    histbook_raw[name] = histogram;

    // Word occurances in all images
    for (size_t k{0}; k < histogram_length; k++) {
      if (histogram.at(k) > 0)
        word_occurances.at(k) += 1;
    }
  }
  histbook_size = histbook_raw.size();
}

std::vector<double> HistBook::TF_IDF_(const std::vector<int> &hist) {
  double word_count = std::accumulate(hist.begin(), hist.end(), 0);
  std::vector<double> histogram(histogram_length, 0);
  for (size_t k{0}; k < histogram_length; k++) {
    // TF-IDF formula
    histogram.at(k) = std::max(
        0.0,
        ((double)hist.at(k) / word_count) *
            (log((double)histbook_size /
                 word_occurances.at(k)))); // TODO:: Looks problematic as
                                           // word_occurances.at(k) can be zero
  }
  return histogram;
}

std::vector<std::string>
HistBook::KNMatcher_(const std::vector<double> &query_hist, const int k) {
  if (!histbook.size())
    std::cout << "ERROR: Unable to load HistBook" << std::endl;

  std::vector<double> cossim;
  std::vector<std::string> names;

  double norm_y = sqrt(std::inner_product(query_hist.begin(), query_hist.end(),
                                          query_hist.begin(), 0.0));

  for (const auto &[name, hist] : histbook) {
    double norm_x =
        sqrt(std::inner_product(hist.begin(), hist.end(), hist.begin(), 0.0));

    double numerator =
        std::inner_product(hist.begin(), hist.end(), query_hist.begin(), 0.0);
    double denominator = norm_x * norm_y;

    // Cosine Similarity -> best match should be close to zero; worst should be
    // close to 1
    double cosine_similarity = 1.0 - (numerator / denominator);
    cossim.emplace_back(cosine_similarity);
    names.emplace_back(name);
  }

  std::vector<std::string> kmatches;
  for (size_t i{0}; i < k; i++) {
    auto closest_match = std::min_element(cossim.begin(), cossim.end());
    auto idx = std::distance(cossim.begin(), closest_match);
    kmatches.emplace_back(names.at(idx));
    cossim.erase(cossim.begin() + idx);
    names.erase(names.begin() + idx);
  }
  return kmatches;
}

std::vector<int> HistBook::computeHist(const cv::Mat &image) {
  sift.detectAndExtract(image);
  cv::Mat des = sift.getDescriptors();
  std::vector<int> histogram = computeHist_(des);
  return histogram;
}

std::vector<int> HistBook::computeHist(const std::filesystem::path &name) {
  const cv::Mat image = cv::imread(name, cv::IMREAD_COLOR);
  std::vector<int> histogram = computeHist(image);
  return histogram;
}

void HistBook::displayHist(const std::vector<int> &hist) {
  int max = *max_element(hist.begin(), hist.end());
  auto row = max;
  for (int i{0}; i <= max; i++) {
    for (const auto &count : hist) {
      if (count >= row)
        std::cout << "*";
      else
        std::cout << " ";
    }
    std::cout << "\n";
    row -= 1;
  }
}

void HistBook::displayHist(const std::vector<double> &hist) {
  std::vector<int> histogram(hist.size(), 0);
  int k = 0;
  for (const auto &bin : hist) {
    histogram.at(k) = round(bin * 1000);
    k++;
  }
  displayHist(histogram);
}

void HistBook::displayHist(const cv::Mat &image) {
  std::vector<int> histogram = computeHist(image);
  displayHist(histogram);
}

void HistBook::displayHist(const std::filesystem::path &name) {
  const cv::Mat image = cv::imread(name, cv::IMREAD_COLOR);
  displayHist(image);
}

void HistBook::generate(const std::filesystem::path &image_ext,
                        const std::string &suffix) {
  computeHistAll_(image_ext, suffix);
  for (auto &[name, hist] : histbook_raw) {
    std::vector<double> histogram = TF_IDF_(hist);
    histbook[name] = histogram;
  }
}

void HistBook::save(const std::filesystem::path &name,
                    const std::string &suffix) {
  auto filename = name;
  auto path = data_path;
  if (!filename.has_extension())
    filename += ".txt";
  else if (filename.extension() != ".txt")
    filename = (filename.stem()) += ".txt";
  path /= filename;

  std::ofstream out_file{path.c_str()};
  out_file << "word_occurances";
  for (const auto &bin : word_occurances) {
    out_file << " " << bin;
  }
  out_file << "\n";

  for (const auto &[name, hist] : histbook) {
    out_file << name;
    for (const auto &val : hist)
      out_file << " " << val;
    out_file << "\n";
  }
  out_file.close();
}

std::map<std::string, std::vector<double>>
HistBook::load(const std::filesystem::path &name, const std::string &suffix) {
  std::map<std::string, std::vector<double>> loaded_histbook;

  auto filename = name;
  if (!filename.has_extension())
    filename += ".txt";
  else if (filename.extension() != ".txt")
    filename = (filename.stem()) += ".txt";

  auto path = data_path;
  path /= filename;

  std::ifstream in_file{path.c_str()};
  std::string line;
  std::string delimiter = " ";

  // Get word_occurances stored in first line
  std::getline(in_file, line);
  std::string identifier = line.substr(0, line.find(delimiter));
  line.erase(0, identifier.length() + delimiter.length());

  size_t pos = 0;
  int count = 0;
  std::string val;
  while ((pos = line.find(delimiter)) != std::string::npos) {
    val = line.substr(0, pos);
    word_occurances.at(count) = std::stoi(val);
    line.erase(0, pos + delimiter.length());
    count++;
  }

  while (std::getline(in_file, line)) {
    std::string name = line.substr(0, line.find(delimiter));
    line.erase(0, name.length() + delimiter.length());

    std::vector<double> hist;
    while ((pos = line.find(delimiter)) != std::string::npos) {
      val = line.substr(0, pos);
      hist.emplace_back(std::stod(val));
      line.erase(0, pos + delimiter.length());
    }

    loaded_histbook[name] = hist;
  }
  in_file.close();

  // Setters
  histbook = loaded_histbook;
  histbook_size = histbook.size();

  return loaded_histbook;
}

std::vector<std::string> HistBook::KNMatcher(const cv::Mat &query_image,
                                             const int &k) {
  std::vector<int> histogram = computeHist(query_image);
  std::vector<double> tfidf_hist = TF_IDF_(histogram);
  std::vector<std::string> kmatches = KNMatcher_(tfidf_hist, k);
  return kmatches;
}

std::vector<std::string>
HistBook::KNMatcher(const std::filesystem::path &filename, const int &k) {
  auto file = filename;
  if (filename.filename() == filename) {
    auto path = data_path;
    file = (path /= filename);
  }
  cv::Mat query_image = cv::imread(file);
  std::vector<int> histogram = computeHist(query_image);
  std::vector<double> tfidf_hist = TF_IDF_(histogram);
  std::vector<std::string> kmatches = KNMatcher_(tfidf_hist, k);
  return kmatches;
}