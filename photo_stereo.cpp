
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <windows.h>
#include <utils.h>

namespace fs = std::filesystem;


int main() {
    fs::path p = fs::current_path() / "images" / "cat";
    //fs::path p = fs::current_path() / "images" / "frog";
    //fs::path p = fs::current_path() / "images" / "lizard";
    std::cout << "Image directory: " << p.string() << "\n";
    if (!fs::exists(p)) {
        std::cerr << "directory not found: " << p << "\n";
        return 1;
    }else if (!fs::is_directory(p)) {
        std::cerr << "Input is not a directory: " << p << "\n";
        return 1;
    }

    std::vector<fs::path> image_paths = get_files_in_folder(p, ".png");
    
    int nr_ims = image_paths.size();
    std::vector <cv::Mat> images;
    images.reserve(nr_ims);

    for (int i = 0; i < nr_ims; i++) {
        // Reading bytes and imdecode (bypasses fopen path issues)
        std::vector<uchar> buf;
        bool opened = try_ifstream(image_paths[i], buf);
        if (opened) {
            cv::Mat im = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
            if (im.empty()) {
                std::cerr << "cv::imdecode failed for: " << image_paths[i].string() << "\n";
                continue;
            }
            images.push_back(std::move(im));
        }
    }

    if (images.empty()) {
        std::cerr << "No images successfully loaded.\n";
        return 1;
    }

	int rows = images[0].rows;
	int cols = images[0].cols;
	int pixels = rows * cols;
	cv::Mat I = cv::Mat(pixels, nr_ims, CV_64F);
    for (int i = 0; i < nr_ims; i++) {
        cv::Mat col = images[i].reshape(1, pixels);

        // Convert to CV_64F directly into the preallocated column of I
        col.convertTo(I.col(i), CV_64F);
    }

    cv::Mat U, S, V;
    cv::SVD::compute(I, S, U, V);

    cv::Mat Uhat = U.colRange(0, 3);
    cv::Mat Shat = S.rowRange(0, 3);
    cv::Mat Vhat = V.rowRange(0, 3);
    cv::Mat Ssqrt = cv::Mat::zeros(3, 3, CV_64F);;
    Ssqrt.at<double>(0, 0) = std::sqrt(Shat.at<double>(0, 0));
    Ssqrt.at<double>(1, 1) = std::sqrt(Shat.at<double>(1, 0));
    Ssqrt.at<double>(2, 2) = std::sqrt(Shat.at<double>(2, 0));
    cv::Mat Nhat = Uhat * Ssqrt;
    cv::Mat Lhat = Ssqrt * Vhat;


    cv::Mat LDir = readSpaceSeparatedMatrix(p / "light_directions.txt");
    cv::Mat A;
    cv::invert(solveLeftLeastSquares(LDir, Lhat), A, cv::DECOMP_SVD);
    cv::Mat N = Nhat * A;

    cv::Mat N_shape = cv::Mat::zeros(N.rows, N.cols, CV_64F);
    cv::Mat xyz = cv::Mat::zeros(N.rows, N.cols, CV_64F);

    for (int i = 0; i < pixels; i++) {
        cv::Mat glow = cv::Mat::zeros(1, nr_ims, CV_64F);
        for (int j = 0; j < nr_ims; j++) {
            //std::cout << I.at<double>(i, j) << "\n";
            if (I.at<double>(i, j) > 20) {
                glow.at<double>(0, j) = 1;
            }           
        }

        if (cv::sum(glow)[0] > 10) {
            N_shape.row(i) = N.row(i) / cv::norm(N.row(i), cv::NORM_L2);
            xyz.at<double>(i, 1) = i % rows;
            int y = i / rows;
            xyz.at<double>(i, 2) = y;
        }
    }

    cv::Mat xyz_3_ch = xyz.reshape(3, rows); 
    std::vector<cv::Mat> channels;
    cv::split(xyz_3_ch, channels); // channels[0]=x, [1]=y, [2]=z
	cv::Mat &xMat = channels[0];
	cv::Mat &yMat = channels[1];
	cv::Mat &zMat = channels[2];

    cv::Mat N_3_ch = N_shape.reshape(3, rows);
    cv::split(N_3_ch, channels); // channels[0]=x, [1]=y, [2]=z
	cv::Mat &x_N_shape = channels[0];
	cv::Mat &y_N_shape = channels[1];
	cv::Mat &z_N_shape = channels[2];

	cv::Mat shape1 = cv::Mat::zeros(rows, cols, CV_64F);
	cv::Mat shape2 = cv::Mat::zeros(rows, cols, CV_64F);
	cv::Mat shape3 = cv::Mat::zeros(rows, cols, CV_64F);
	cv::Mat abs_x_N_shape, abs_y_N_shape, abs_z_N_shape;
	cv::absdiff(x_N_shape, cv::Scalar(0), abs_x_N_shape);
    cv::absdiff(y_N_shape, cv::Scalar(0), abs_y_N_shape);
    cv::absdiff(z_N_shape, cv::Scalar(0), abs_z_N_shape);
	cv::threshold(abs_x_N_shape, shape1, 0.0, 1.0, cv::THRESH_BINARY);
	cv::threshold(abs_y_N_shape, shape2, 0.0, 1.0, cv::THRESH_BINARY);
	cv::threshold(abs_z_N_shape, shape3, 0.0, 1.0, cv::THRESH_BINARY);
    std::vector<cv::Point> a;
    cv::findNonZero(shape1, a);
    cv::findNonZero(shape2, a);
    cv::findNonZero(shape3, a);
	cv::Mat shape_sum = shape1 + shape2 + shape3;
    cv::Mat shape;
	cv::threshold(shape_sum, shape, 1.0, 1.0, cv::THRESH_BINARY);
    std::vector<cv::Point> non_zero;
    cv::Mat visited = cv::Mat::zeros(rows, cols, CV_8U);
	cv::findNonZero(shape, non_zero);
    if (non_zero.empty()) {
        std::cerr << "No non-zero shape pixels found.\n";
        // handle gracefully: skip, continue, or abort
    }
    else {
        zMat.at<double>(non_zero[0]) = 0.001;
        visited.at<uchar>(non_zero[0]) = 1;
    }

    bool done = false;
    int count = 0;

    while (!done) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if (j + 1 < cols && visited.at<uchar>(i, j + 1) == 0 && shape.at<double>(i, j + 1) != 0) {
					visited.at<uchar>(i, j + 1) = 1;
					zMat.at<double>(i, j + 1) = zMat.at<double>(i, j) + (y_N_shape.at<double>(i, j) / z_N_shape.at<double>(i, j));
				}
                else if (i + 1 < rows && visited.at<uchar>(i + 1, j) == 0 && shape.at<double>(i + 1, j) != 0) {
					zMat.at<double>(i + 1, j) = zMat.at<double>(i, j) + (x_N_shape.at<double>(i, j) / z_N_shape.at<double>(i, j));
					visited.at<uchar>(i + 1, j) = 1;
				}
                else if (j - 1 >= 0 && visited.at<uchar>(i, j - 1) == 0 && shape.at<double>(i, j - 1) != 0 ) {
                    zMat.at<double>(i, j - 1) = zMat.at<double>(i, j) - (y_N_shape.at<double>(i, j) / z_N_shape.at<double>(i, j));
                    visited.at<uchar>(i, j - 1) = 1;
                }
                else if (i - 1 >= 0 && visited.at<uchar>(i - 1, j) == 0 && shape.at<double>(i - 1, j) != 0) {
                    zMat.at<double>(i - 1, j) = zMat.at<double>(i, j) - (x_N_shape.at<double>(i, j) / z_N_shape.at<double>(i, j));
                    visited.at<uchar>(i - 1, j) = 1;
                }
            }
		}

		// All pixels visited
        if (cv::countNonZero(zMat)) {
			done = true;
        }
		// Limit iterations to prevent infinite loop in case of isolated pixels
        count++;
        if (count > 2000) {
            done = true;
		}
    }

	xyz.col(2) = zMat.reshape(1, pixels);
	cv::Mat object_3_ch = xyz.reshape(3, rows);

    showDepth(object_3_ch);
	show3D(object_3_ch, "Reconstructed 3D Shape");

    cv::Mat gray = images[0];
    cv::Mat edges;
    cv::imshow("Test Window Name", images[0]);
    cv::waitKey(5000);
    //Sleep(5000);
    std::cout << "Processed first image.\n";

    return 0;

}
