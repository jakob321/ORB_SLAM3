#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <opencv2/core/core.hpp>

// Include the ORB-SLAM3 header (adjust the include path as needed)
#include "System.h"

namespace py = pybind11;

// void LoadImages(const string &strImagePath, const string &strPathTimes,
//                 vector<string> &vstrImages, vector<double> &vTimeStamps);
void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
    vector<double> &vTimestamps);


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
    if (voc_file.empty() || settings_file.empty())
    {
        return "Not enough parameters provided. Please supply both voc_file and settings_file.";
    }

    vector<string> vstrImageFilenames;
    vector<double> vTimestampsCam;

    cout << "Loading images..." << endl;
    // LoadImages(string(imageFolder), string(timestampFile), vstrImageFilenames, vTimestampsCam);
    LoadImages(string(imageFolder), vstrImageFilenames, vTimestampsCam);
    cout << "LOADED!" << endl;
    cout << vstrImageFilenames[100] << endl;
    cout << vTimestampsCam[0] << vTimestampsCam[1] << vTimestampsCam[2] << vTimestampsCam[3] <<endl;

    int nImages = vstrImageFilenames.size();
    int tot_images = nImages;
    cout.precision(17);

    cout << "Total images loaded: " << tot_images << endl; // Debug print the number of images loaded

    // Create the ORB-SLAM3 system.
    // Here we use the MONOCULAR mode and enable viewer (true)
    ORB_SLAM3::System SLAM(voc_file, settings_file, ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);
    

    //return 0;
    // ---------------------------------------------
    // Main loop
    // ---------------------------------------------
    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++)
    {

        // Read image from file
        std::cout<<vstrImageFilenames[ni]<<std::endl;
        im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestampsCam[ni];

        if (imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        // cout << "tframe = " << tframe << endl;
        Sophus::SE3f Tcw = SLAM.TrackMonocular(im, tframe);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        Sophus::SE3f Twc = Tcw.inverse(); // Now Twc is the pose of the camera in the world coordinate frame.
        Eigen::Vector3f t = Twc.translation();
        Eigen::Quaternionf q(Twc.rotationMatrix());

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestampsCam[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestampsCam[ni - 1];
        
        std::cout << "loop iter: " << ni << std::endl;
        // std::cout << "T: " << T << std::endl;
        // std::cout << "ttrack: " << ttrack << std::endl;

        if (ttrack < T)
        {
            // std::cout << "usleep: " << (dT-ttrack) << std::endl;
            usleep((T - ttrack) * 1e6); // 1e6
        }
    }

    // Shut down the system (this stops all threads).
    SLAM.Shutdown();

    // Return a simple message confirming that the system was initialized.
    return "ORB-SLAM3 system initialized and shut down successfully.";
}

// void LoadImages(const string &strImagePath, const string &strPathTimes,
//                 vector<string> &vstrImages, vector<double> &vTimeStamps)
// {
//     ifstream fTimes;
//     fTimes.open(strPathTimes.c_str());
//     vTimeStamps.reserve(5000);
//     vstrImages.reserve(5000);
//     while (!fTimes.eof())
//     {
//         string s;
//         getline(fTimes, s);
//         if (!s.empty())
//         {
//             stringstream ss;
//             ss << s;
//             vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
//             double t;
//             ss >> t;
//             vTimeStamps.push_back(t * 1e-9);
//         }
//     }
// }

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    int nTimes = 0;
    double fps = 30;
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/timestamps.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            nTimes++;
        }
    }

    string strPrefixLeft = strPathToSequence + "/data/";

    //const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(10) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
        vTimestamps.push_back(i/fps);
    }
}


// Function that processes a NumPy array efficiently
py::array_t<double> multiply_array(py::array_t<double> input_array) {
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
    for (size_t i = 0; i < size; i++) {
        ptr_out[i] = ptr[i] * 2;
    }

    return result;
}

// Define the module and bind functions here (single module definition)
PYBIND11_MODULE(orbslam3, m) {
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
}
