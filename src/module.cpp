#include <map>
#include <utility>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "DVSSimulator.h"

using std::vector;
using std::cerr;
using std::endl;
using std::max;
using std::min;
using std::invalid_argument;

namespace py = pybind11;

PYBIND11_MODULE(simulator, m) {
  m.doc() = "module to convert a sequence of images to an event stream";
  py::class_<DVSSimulator>(m, "DVSSimulator")
    .def(py::init<py::EigenDRef<MatrixXuc> &, uint64_t, float>(),
        "Constructs simulator with an equal sensitivity value for all pixels\n\n"
        "Parameters\n"
        "----------\n"
        "img: 2D ndarray of unsigned chars\n"
        "\tAn inital image\n"
        "timestamp: int\n"
        "\tA timestamp of the initial image\n"
        "C: float\n"
        "\tA sensitivity value"
        )
    .def(py::init<py::EigenDRef<MatrixXuc> &, uint64_t, py::EigenDRef<Eigen::ArrayXXf> &>(),
        "Constructs simulator with a matrix of sensitivity values\n\n"
        "Parameters\n"
        "----------\n"
        "img: 2D ndarray of unsigned chars\n"
        "\tAn inital image\n"
        "timestamp: int\n"
        "\tA timestamp of the initial image\n"
        "C: 2D ndarray of float32\n"
        "\tA sensitivity matrix, where C[i, j] specifies sensitivity of pixel at (i, j)"
        )
    .def("update", &DVSSimulator::update, py::call_guard<py::gil_scoped_release>(),
        "Generates events between the last observed and the current frame\n\n"
        "Parameters\n"
        "----------\n"
        "img: 2D ndarray of unsigned chars\n"
        "\tA current image\n"
        "timestamp: int\n"
        "\tA timestamp of the input image\n\n"
        "Returns\n"
        "-------\n"
        "dict\n"
        "\tThe constructed events in a form of dictionary with keys \"timestamps\", "
        "\"x_positions\", \"y_positions\" and \"polarities\""
        )
    .def_property_readonly("timestamp", &DVSSimulator::get_timestamp,
        "The last observed timestamp")
    .def_property_readonly("C", &DVSSimulator::get_C,
        "The sensitivity matrix of DVS")
    ;
}
