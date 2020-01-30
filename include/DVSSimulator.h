#include <vector>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

typedef Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXuc;
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, 1> VectorXull;
typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> VectorXui;
typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;

namespace py = pybind11;

class DVSSimulator {
  public:

    /**
     * Constructor with a single threshold for all pixels
     *
     * \param in_img a grayscale image
     * \param in_timestamp a timestamp of the input image
     * \param in_C a threshold used by DVS to generate new event
     */
    DVSSimulator(py::EigenDRef<MatrixXuc> &in_img, uint64_t in_timestamp, float in_C);
    /**
     * Constructor with a matrix of pixel thresholds
     *
     * \param in_img a grayscale image
     * \param in_timestamp a timestamp of the input image
     * \param in_C a matrix of thresholds for each pixel.
     * It has to have size of the input image
     */
    DVSSimulator(py::EigenDRef<MatrixXuc> &in_img, uint64_t in_timestamp,
        py::EigenDRef<Eigen::ArrayXXf> &in_C);
    virtual ~DVSSimulator() = default;

    py::dict update(py::EigenDRef<MatrixXuc> &in_img, uint64_t in_timestamp);

    uint64_t get_timestamp() const {
      return m_timestamp;
    }

    const Eigen::ArrayXXf &get_C() const {
      return m_C;
    }

  protected:
    /// the last seen log-image
    Eigen::ArrayXXf m_img;
    /// the matrix of the reference values in the form of levels archived
    /// on the previous time stamp
    Eigen::ArrayXXf m_reference;
    /// timestamp corresponding to the last seen log-image
    uint64_t m_timestamp;
    /// sensitivity
    Eigen::ArrayXXf m_C;

    /** converts the intput grayscale image to the log-image

     \param[in] in_img an input grayscale image
     \return an log-image
     */
    static inline Eigen::ArrayXXf safe_log(py::EigenDRef<MatrixXuc> &in_img);
    
    template<typename T>
    static Eigen::Matrix<T, Eigen::Dynamic, 1> rearrange(
        const std::vector<T> &data,
        const std::vector<size_t> &idx_array) {
      const auto n = data.size();
      Eigen::Matrix<T, Eigen::Dynamic, 1> res(n);
      auto ptr = res.data();
      for (auto i = 0u; i < n; ++i)
        ptr[i] = data[idx_array[i]];
      return res;
    }

    std::tuple<VectorXull, VectorXui, VectorXui, VectorXb> update_log(
        Eigen::ArrayXXf &in_img, uint64_t in_timestamp);
  private:
};
