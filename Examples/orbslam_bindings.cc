// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <opencv2/core/core.hpp>
// #include "System.h"  // ORB_SLAM3 header (adjust include path as needed)

// namespace py = pybind11;

// // Helper function: Convert a NumPy array (assumed to be uint8)
// // to an OpenCV Mat. This version works for 2D (grayscale) or 3D (color) images.
// cv::Mat numpy_uint8_to_cv_mat(py::array_t<uint8_t>& input) {
//     py::buffer_info buf = input.request();
//     if (buf.ndim < 2 || buf.ndim > 3)
//         throw std::runtime_error("Input array must have 2 or 3 dimensions");
    
//     int rows = buf.shape[0];
//     int cols = buf.shape[1];
//     int channels = (buf.ndim == 3 ? buf.shape[2] : 1);
    
//     int type = (channels == 1 ? CV_8UC1 : CV_8UC3);
//     // Create a Mat header pointing to the buffer data.
//     cv::Mat mat(rows, cols, type, (unsigned char*)buf.ptr);
//     // Clone to ensure the data is owned by the Mat.
//     return mat.clone();
// }

// PYBIND11_MODULE(orbslam3, m) {
//     m.doc() = "pybind11 binding for ORB-SLAM3";

//     // Expose the sensor type enum.
//     py::enum_<ORB_SLAM3::System::eSensor>(m, "Sensor")
//         .value("MONOCULAR", ORB_SLAM3::System::MONOCULAR)
//         .value("STEREO", ORB_SLAM3::System::STEREO)
//         .value("RGBD", ORB_SLAM3::System::RGBD)
//         .export_values();

//     // Bind the ORB_SLAM3::System class.
//     py::class_<ORB_SLAM3::System>(m, "System")
//         .def(py::init<const std::string&, const std::string&, ORB_SLAM3::System::eSensor, bool>(),
//              py::arg("vocabulary_file"),
//              py::arg("settings_file"),
//              py::arg("sensor_mode"),
//              py::arg("use_viewer") = true,
//              "Initialize the ORB-SLAM3 system.")
//         .def("track_monocular",
//              [](ORB_SLAM3::System &self, py::array_t<uint8_t> image, double timestamp) {
//                  cv::Mat img = numpy_uint8_to_cv_mat(image);
//                  self.TrackMonocular(img, timestamp);
//              },
//              py::arg("image"), py::arg("timestamp"),
//              "Process a monocular image frame.")
//         .def("shutdown",
//              &ORB_SLAM3::System::Shutdown,
//              "Shutdown the SLAM system.")
//           .def("save_trajectory_euroc",
//                static_cast<void (ORB_SLAM3::System::*)(const std::string&)>(
//                     &ORB_SLAM3::System::SaveTrajectoryEuRoC
//                ),
//                py::arg("filename"),
//                "Save the camera trajectory in EuRoC format.")
//         .def("save_keyframe_trajectory_euroc",
//              &ORB_SLAM3::System::SaveKeyFrameTrajectoryEuRoC,
//              py::arg("filename"),
//              "Save the keyframe trajectory in EuRoC format.")
//         .def("change_dataset",
//              &ORB_SLAM3::System::ChangeDataset,
//              "Change dataset for processing.")
//         .def("get_image_scale",
//              &ORB_SLAM3::System::GetImageScale,
//              "Get the image scaling factor.");
// }

// orbslam3_bindings.cpp
// orbslam3_bindings.cpp
#include <pybind11/pybind11.h>
#include <string>

// Include the ORB-SLAM3 header (adjust the include path as needed)
#include "System.h"

namespace py = pybind11;

/**
 * Minimal function that creates an ORB-SLAM3 system, then immediately shuts it down.
 * If either parameter is empty, it returns an error message.
 *
 * @param voc_file Path to the vocabulary file.
 * @param settings_file Path to the settings file.
 * @return A confirmation message or error message if parameters are missing.
 */
std::string run_orb_slam3(const std::string &voc_file = "", const std::string &settings_file = "") {
    if(voc_file.empty() || settings_file.empty()){
        return "Not enough parameters provided. Please supply both voc_file and settings_file.";
    }
    
    // Create the ORB-SLAM3 system.
    // Here we use the MONOCULAR mode and enable viewer (true) for simplicity.
    ORB_SLAM3::System slam(voc_file, settings_file, ORB_SLAM3::System::MONOCULAR, true);

    // For this minimal example we do nothing more.

    // Shut down the system (this stops all threads).
    slam.Shutdown();

    // Return a simple message confirming that the system was initialized.
    return "ORB-SLAM3 system initialized and shut down successfully.";
}

PYBIND11_MODULE(orbslam3, m) {
    m.doc() = "Minimal Python bindings for ORB-SLAM3";
    m.def("run_orb_slam3", &run_orb_slam3,
          "Initialize and shutdown the ORB-SLAM3 system minimally. "
          "Requires voc_file and settings_file as parameters.",
          py::arg("voc_file") = "", py::arg("settings_file") = "");
}



// #include <pybind11/pybind11.h>
// #include <string>

// namespace py = pybind11;

// std::string hello() {
//     return "Hello from ORB-SLAM3 bindings!";
// }

// PYBIND11_MODULE(orbslam3, m) {
//     m.def("hello", &hello, "A function that returns a greeting");
// }

