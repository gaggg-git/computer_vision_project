#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <windows.h>

namespace fs = std::filesystem;


std::vector<fs::path> get_files_in_folder(const fs::path& folder, const std::string& file_ext = {});

bool try_ifstream(const fs::path& p, std::vector<uchar>& buf);

cv::Mat readSpaceSeparatedMatrix(const std::filesystem::path& path);

cv::Mat solveLeftLeastSquares(const cv::Mat& A, const cv::Mat& B);

void showDepth(const cv::Mat& xyz_3ch);

void show3D(cv::Mat& xyz_3ch, const std::string& window_name = "3D Points");
