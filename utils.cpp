#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <windows.h>
#include <opencv2/viz.hpp>

namespace fs = std::filesystem;

//get all file paths in a folder (with specific extension)
std::vector<fs::path> get_files_in_folder(const fs::path& folder, const std::string& file_ext = {}) {
    std::vector<fs::path> files;
    try {
        if (!fs::exists(folder) || !fs::is_directory(folder)) return files;
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.is_regular_file()) {
                if (file_ext.empty() || entry.path().extension() == file_ext)
                    files.push_back(entry.path());
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "filesystem error: " << e.what() << "\n";
    }
    return files;
}

bool try_ifstream(const fs::path& p, std::vector<uchar>& buf) {
    std::ifstream f(p, std::ios::binary);
    if (!f.is_open()) return false;
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    if (sz <= 0) return false;
    f.seekg(0, std::ios::beg);
    buf.resize(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return true;
}

cv::Mat readSpaceSeparatedMatrix(const std::filesystem::path& path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    std::string line;
    std::vector<std::vector<double>> rows;
    size_t cols = 0;
    size_t lineNo = 0;

    while (std::getline(ifs, line)) {
        ++lineNo;
        // trim leading spaces
        size_t pos = line.find_first_not_of(" \t\r\n");
        if (pos == std::string::npos) continue;           // empty line
        if (line[pos] == '#') continue;                   // commented line

        std::istringstream iss(line);
        std::vector<double> values;
        double v;
        while (iss >> v) values.push_back(v);

        if (values.empty()) continue;                     // handle lines with only whitespace

        if (cols == 0) cols = values.size();
        else if (values.size() != cols) {
            throw std::runtime_error("Inconsistent column count at line " + std::to_string(lineNo));
        }

        rows.push_back(std::move(values));
    }

    if (rows.empty()) return cv::Mat(); // empty matrix

    int rowsCount = static_cast<int>(rows.size());
    int colsCount = static_cast<int>(cols);
    cv::Mat matrix(rowsCount, colsCount, CV_64F);

    for (int r = 0; r < rowsCount; ++r) {
        for (int c = 0; c < colsCount; ++c) {
            matrix.at<double>(r, c) = static_cast<double>(rows[r][c]);
        }
    }

    return matrix;
}

// Solves A = X * B for X (least-squares)
cv::Mat solveLeftLeastSquares(const cv::Mat& A, const cv::Mat& B)
{
    if (A.empty() || B.empty()) throw std::invalid_argument("Empty matrix");
    if (A.cols != B.cols) throw std::invalid_argument("Column mismatch: A.cols must equal B.cols");

    cv::Mat Af, Bf;
    A.convertTo(Af, CV_64F);
    B.convertTo(Bf, CV_64F);

    // Method 1: pseudoinverse via SVD (concise)
    cv::Mat ps_inv_B;
    cv::invert(Bf, ps_inv_B, cv::DECOMP_SVD);
    cv::Mat X = Af * ps_inv_B; // (m x n) * (n x p) -> m x p
    X.convertTo(X, A.type());
    return X;
}

void showDepth(const cv::Mat& xyz_3ch) {
    CV_Assert(xyz_3ch.channels() == 3);
    cv::Mat z;
    std::vector<cv::Mat> ch;
    cv::split(xyz_3ch, ch); // ch[2] is z if xyz order
    ch[2].convertTo(z, CV_32F);

    // Mask invalid z (<=0) to avoid skewing normalization
    cv::Mat mask = (z <= 0);
    double minVal, maxVal;
    cv::minMaxLoc(z, &minVal, &maxVal, nullptr, nullptr, ~mask);

    cv::Mat depth8;
    z.setTo(minVal, mask);
    z.convertTo(depth8, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    cv::applyColorMap(depth8, depth8, cv::COLORMAP_JET);
    depth8.setTo(cv::Scalar(0, 0, 0), mask);
    cv::imshow("Depth", depth8);
    cv::waitKey(0);
}

void show3D(cv::Mat& xyz_3ch, const std::string& window_name = "3D Points") {
    // assume xyz_3ch is rows x cols with 3 channels (CV_64F or CV_32F)
    int rows = xyz_3ch.rows, cols = xyz_3ch.cols;

    cv::Mat xyz32;
    xyz_3ch.convertTo(xyz32, CV_32F);            // viz expects float
    cv::Mat cloud = xyz32.reshape(1, rows * cols); // Nx3 (x,y,z)

    // remove invalid points (z <= 0)
    std::vector<int> validIdx;
    validIdx.reserve(cloud.rows);
    for (int i = 0; i < cloud.rows; ++i) {
        float z = cloud.at<cv::Vec3f>(i)[2];
        if (z > 0 && std::isfinite(z)) validIdx.push_back(i);
    }
    cv::Mat cloudValid((int)validIdx.size(), 3, CV_32F);
    for (size_t i = 0; i < validIdx.size(); ++i) {
        cloud.row(validIdx[i]).copyTo(cloudValid.row((int)i));
    }

    cv::viz::WCloud wcloud(cloudValid);
    cv::viz::Viz3d viz("Point Cloud");
    viz.showWidget("cloud", wcloud);
    viz.spin(); // or spinOnce loop for interactivity
}