#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h> // Optional: For Eigen conversion
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <unistd.h> // for usleep
#include <sstream>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>

// Include the ORB-SLAM3 header (adjust the include path as needed)
#include "System.h"

namespace py = pybind11;
using std::cout;
using std::endl;
using std::string;
using std::vector;

// ================= Global Variables for SLAM Thread and Camera Pose =================

// Global variables to store the latest camera pose as a 4x4 matrix.
// We use a mutex to ensure that the pose is updated/read safely.
std::mutex camPoseMutex;
Eigen::Matrix4f latestCamPose = Eigen::Matrix4f::Identity();


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    int nTimes = 0;
    double fps = 100;
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/timestamps.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            nTimes++;
        }
    }

    string strPrefixLeft = strPathToSequence + "/data/";

    // const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for (int i = 0; i < nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(10) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
        vTimestamps.push_back(i / fps);
    }
}

/**
 * Minimal function that creates an ORB-SLAM3 system, then immediately shuts it down.
 * If either parameter is empty, it returns an error message.
 *
 * @param voc_file Path to the vocabulary file.
 * @param settings_file Path to the settings file.
 * @return A confirmation message or error message if parameters are missing.
 */
std::string run_orb_slam3(const std::string &voc_file = "",
                          const std::string &settings_file = "",
                          const std::string &imageFolder = "",
                          const std::string &timestampFile = "")
{
    py::gil_scoped_release release;
    if (voc_file.empty() || settings_file.empty())
    {
        std::cerr << "Not enough parameters provided. Please supply both voc_file and settings_file." << std::endl;
    }

    vector<string> vstrImageFilenames;
    vector<double> vTimestampsCam;

    cout << "Loading images..." << endl;
    LoadImages(imageFolder, vstrImageFilenames, vTimestampsCam);
    cout << "Images loaded!" << endl;
    int nImages = vstrImageFilenames.size();
    cout << "Total images loaded: " << nImages << endl;

    // Create the ORB-SLAM3 system in MONOCULAR mode (viewer enabled).
    ORB_SLAM3::System SLAM(voc_file, settings_file, ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    vector<float> vTimesTrack(nImages, 0.0f);

    cv::Mat im;
    for (int ni = 0; ni < (nImages - 10); ni++)
    {
        // cout << "Processing image: " << vstrImageFilenames[ni] << endl;
        im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestampsCam[ni];

        if (imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        auto t1 = std::chrono::steady_clock::now();
        Sophus::SE3f Tcw = SLAM.TrackMonocular(im, tframe);
        auto t2 = std::chrono::steady_clock::now();
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = static_cast<float>(ttrack);

        // Get the camera pose in the world coordinate frame.
        Sophus::SE3f Twc = Tcw.inverse();

        // Update the global camera pose.
        {
            std::lock_guard<std::mutex> lock(camPoseMutex);
            // Convert to a 4x4 matrix of type float.
            latestCamPose = Twc.matrix().cast<float>();
        }

        double T = 0;
        if (ni < nImages - 1)
            T = vTimestampsCam[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestampsCam[ni - 1];

        // cout << "Loop iteration: " << ni << ", ttrack: " << ttrack
        //     << ", sleep time: " << (T - ttrack) << endl;
        if (ttrack < T)
        {
            usleep((T - ttrack) * 1e6); // Sleep in microseconds.
        }
    }

    SLAM.Shutdown();
    return "success";
}

// Example function that processes a NumPy array efficiently
py::array_t<double> multiply_array(py::array_t<double> input_array)
{
    // Get buffer information (avoids copying)
    py::buffer_info buf = input_array.request();

    // Get pointer to data
    double *ptr = static_cast<double *>(buf.ptr);
    size_t size = buf.size;

    // Create an output array (same size)
    py::array_t<double> result(buf.size);
    py::buffer_info buf_out = result.request();
    double *ptr_out = static_cast<double *>(buf_out.ptr);

    // Process the array (multiply each element by 2)
    for (size_t i = 0; i < size; i++)
    {
        ptr_out[i] = ptr[i] * 2;
    }

    return result;
}

// Retrieves the latest camera pose as a 4x4 NumPy array.
py::array_t<float> get_camera_pose()
{
    std::lock_guard<std::mutex> lock(camPoseMutex);
    py::array_t<float> result({4, 4});
    auto r = result.mutable_unchecked<2>();
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            r(i, j) = latestCamPose(i, j);
        }
    }
    return result;
}

// Define the module and bind functions here (single module definition)
PYBIND11_MODULE(orbslam3, m)
{
    m.doc() = "Minimal Python bindings for ORB-SLAM3 and NumPy integration";

    // Bind the SLAM function
    m.def("run_orb_slam3", &run_orb_slam3,
          "Initialize and shutdown the ORB-SLAM3 system minimally. "
          "Requires voc_file and settings_file as parameters.",
          py::arg("voc_file") = "",
          py::arg("settings_file") = "",
          py::arg("imageFolder") = "",
          py::arg("timestampFile") = "");

    // Bind the NumPy function
    m.def("multiply_array", &multiply_array,
          "Multiply all elements in a NumPy array by 2.");

    m.def("get_camera_pose", &get_camera_pose,
          "Retrieve the latest camera pose (world coordinate frame) as a 4x4 NumPy array.");
}
