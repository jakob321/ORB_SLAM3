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

// Global containers to hold all camera poses and frame points.
// Global containers to hold the camera poses and per-frame points.
std::vector<Eigen::Matrix4f> allCamPoses;    // Each is 4x4
std::vector<Eigen::MatrixXf> allFramePoints; // Each is 3 x m_i

#include <set> // Needed for std::set

std::vector<Eigen::Vector3f> GetPointCloud(ORB_SLAM3::System &SLAM)
{
    std::vector<Eigen::Vector3f> pointCloud;
    auto atlas = SLAM.GetAtlas();
    int numMaps = atlas->CountMaps();
    std::vector<ORB_SLAM3::Map *> allMaps = atlas->GetAllMaps();

    for (size_t i = 0; i < static_cast<size_t>(numMaps); i++)
    {
        ORB_SLAM3::Map *pMap = allMaps[i];
        if (!pMap)
            continue;

        const std::vector<ORB_SLAM3::MapPoint *> &vpMPs = pMap->GetAllMapPoints();
        const std::vector<ORB_SLAM3::MapPoint *> &vpRefMPs = pMap->GetReferenceMapPoints();
        std::set<ORB_SLAM3::MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

        // Add non-reference map points (skip bad ones and those in the reference set)
        for (size_t j = 0; j < vpMPs.size(); j++)
        {
            if (vpMPs[j]->isBad() || spRefMPs.count(vpMPs[j]))
                continue;
            Eigen::Vector3f pos = vpMPs[j]->GetWorldPos();
            pointCloud.push_back(pos);
        }

        // Add reference map points (skip bad ones)
        for (auto it = spRefMPs.begin(); it != spRefMPs.end(); ++it)
        {
            if ((*it)->isBad())
                continue;
            Eigen::Vector3f pos = (*it)->GetWorldPos();
            pointCloud.push_back(pos);
        }
    }

    return pointCloud;
}

// ================= Global Variables for SLAM Thread and Camera Pose =================

// Global variables to store the latest camera pose as a 4x4 matrix.
// We use a mutex to ensure that the pose is updated/read safely.
std::mutex camPoseMutex;
Eigen::Matrix4f latestCamPose = Eigen::Matrix4f::Identity();
std::vector<ORB_SLAM3::MapPoint *> latestCamPoints;
std::vector<Eigen::Vector3f> allPoints;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    int nTimes = 0;
    double fps = 10;
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
    for (int ni = 0; ni < (nImages - 400); ni++)
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

        {
            std::lock_guard<std::mutex> lock(camPoseMutex);
            // Update current camera pose.
            latestCamPose = Twc.matrix().cast<float>();
            // Append the current camera pose.
            allCamPoses.push_back(latestCamPose);

            // Filter the tracked map points.
            std::vector<ORB_SLAM3::MapPoint *> trackedMapPoints = SLAM.GetTrackedMapPoints();
            std::vector<ORB_SLAM3::MapPoint *> nonNullMapPoints;
            std::copy_if(trackedMapPoints.begin(), trackedMapPoints.end(),
                         std::back_inserter(nonNullMapPoints),
                         [](ORB_SLAM3::MapPoint *p)
                         { return p != nullptr; });
            cout << "Filtered map points: " << nonNullMapPoints.size() << endl;

            // Create a 3 x m matrix for the current frame’s points.
            size_t m = nonNullMapPoints.size();
            Eigen::MatrixXf framePts(3, m);
            for (size_t i = 0; i < m; i++)
            {
                Eigen::Vector3f pos = nonNullMapPoints[i]->mWorldPos;
                framePts(0, i) = pos(0);
                framePts(1, i) = pos(1);
                framePts(2, i) = pos(2);
            }
            // Append the current frame’s points.
            allFramePoints.push_back(framePts);
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

// Retrieves the latest camera keypoints as a 2D NumPy array.
// Each keypoint is represented by [x, y, size, angle, response, octave, class_id]
// py::array_t<float> get_camera_points() {
//     std::lock_guard<std::mutex> lock(camPoseMutex);
//     // Number of keypoints and 7 attributes per keypoint.
//     size_t numPoints = latestCamPoints.size();
//     py::array_t<float> result(py::array::ShapeContainer({static_cast<py::ssize_t>(numPoints), 7}));
//     auto r = result.mutable_unchecked<2>();
//     for (size_t i = 0; i < numPoints; i++) {
//         const cv::KeyPoint& kp = latestCamPoints[i];
//         r(i, 0) = kp.pt.x;
//         r(i, 1) = kp.pt.y;
//         r(i, 2) = kp.size;
//         r(i, 3) = kp.angle;
//         r(i, 4) = kp.response;
//         r(i, 5) = static_cast<float>(kp.octave);
//         r(i, 6) = static_cast<float>(kp.class_id);
//     }
//     return result;
// }

// Retrieves the latest camera pose as a 4x4 NumPy array.
// py::array_t<float> get_camera_pose()
// {
//     std::lock_guard<std::mutex> lock(camPoseMutex);
//     py::array_t<float> result({4, 4});
//     auto r = result.mutable_unchecked<2>();
//     for (size_t i = 0; i < 4; i++)
//     {
//         for (size_t j = 0; j < 4; j++)
//         {
//             r(i, j) = latestCamPose(i, j);
//         }
//     }
//     return result;
// }

py::tuple get_all_data_np() {
    std::lock_guard<std::mutex> lock(camPoseMutex);
    size_t n = allCamPoses.size();

    // Create a shape vector for poses: (4,4,n)
    std::vector<py::ssize_t> pose_shape{4, 4, static_cast<py::ssize_t>(n)};
    py::array_t<float> poses(pose_shape);
    auto poses_buf = poses.mutable_unchecked<3>();
    for (size_t k = 0; k < n; k++) {
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                poses_buf(i, j, k) = allCamPoses[k](i, j);
            }
        }
    }

    // Build a Python list of point arrays (each of shape (3, m))
    py::list points_list;
    for (size_t k = 0; k < allFramePoints.size(); k++) {
        Eigen::MatrixXf &pts = allFramePoints[k];  // pts is 3 x m_k
        int m = pts.cols();
        std::vector<py::ssize_t> pts_shape{3, m};
        py::array_t<float> pts_arr(pts_shape);
        auto pts_buf = pts_arr.mutable_unchecked<2>();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < m; j++) {
                pts_buf(i, j) = pts(i, j);
            }
        }
        points_list.append(pts_arr);
    }

    return py::make_tuple(poses, points_list);
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

    m.def("get_all_data_np", &get_all_data_np,
          "Return a tuple (poses, points_list) where 'poses' is a NumPy array of shape (4,4,n) "
          "and 'points_list' is a list of NumPy arrays (each of shape (3, m_i)).");

// m.def("get_camera_pose", &get_camera_pose,
//       "Retrieve the latest camera pose (world coordinate frame) as a 4x4 NumPy array.");

// m.def("get_camera_points", &get_camera_points,
//   "Retrieve the latest camera keypoints as a 2D NumPy array "
//   "(each row: x, y, size, angle, response, octave, class_id).");
}
