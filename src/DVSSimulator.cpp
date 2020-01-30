#include <DVSSimulator.h>
#include <numeric>
#include <iostream>
#include <stdexcept>

using Eigen::ArrayXXf;
using py::EigenDRef;

using namespace pybind11::literals; // to bring in the `_a` literal

using std::vector;
using std::tuple;
using std::iota;
using std::sort;
using std::swap;
using std::get;
using std::make_tuple;
using std::move;
using std::cerr;
using std::endl;
using std::invalid_argument;

void process_el(vector<uint64_t> &ts, vector<uint32_t> &x,
    vector<uint32_t> &y, vector<bool> &pol, float &crossing, float cur,
    float next, uint64_t c_ts, uint64_t n_ts, float C, uint32_t col, uint32_t row) {
    
  uint64_t dt = n_ts - c_ts;
  float delta = next - cur;
  float step = dt / delta;
  
  bool polarity = next > cur;
  int factor = polarity? +1: -1;
  
  // if polarity
  //   while (crossing + C < next)
  // else
  //   while (crossing - C > next)
  // <=>
  // while (factor * crossing + C < factor * next)
  while (factor * crossing + C <= factor * next) {
    // if polarity
    //   crossing += C
    // else
    //   crossing -= C
    // <=>
    // crossing += factor * C
    crossing += factor * C;
    ts.push_back(c_ts +
           static_cast<uint64_t>(step * (crossing - cur)));
    x.push_back(col);
    y.push_back(row);
    pol.push_back(polarity);
  }
}

DVSSimulator::DVSSimulator(EigenDRef<MatrixXuc> &in_img, uint64_t in_timestamp, float in_C)
: m_timestamp(in_timestamp) {
  m_img = safe_log(in_img);
  m_reference = m_img;
  m_C = ArrayXXf::Constant(m_img.rows(), m_img.cols(), in_C);
}

DVSSimulator::DVSSimulator(EigenDRef<MatrixXuc> &in_img, uint64_t in_timestamp,
    EigenDRef<ArrayXXf> &in_C) :m_timestamp(in_timestamp) {
  m_img = safe_log(in_img);
  m_reference = m_img;
  m_C = in_C;
}

py::dict DVSSimulator::update(EigenDRef<MatrixXuc> &in_img, uint64_t in_timestamp) {
  auto next_img = safe_log(in_img);
  auto res = update_log(next_img, in_timestamp);
  py::gil_scoped_acquire gil;
  return py::dict("timestamps"_a=get<0>(res),
      "x_positions"_a=get<1>(res), "y_positions"_a=get<2>(res),
      "polarities"_a=get<3>(res));
}

inline ArrayXXf DVSSimulator::safe_log(EigenDRef<MatrixXuc> &in_img) {
  float eps = 1e-3;
  ArrayXXf res = in_img.cast<float>();
  res += eps;
  return res.log();
}

tuple<VectorXull, VectorXui, VectorXui, VectorXb> DVSSimulator::update_log(
    ArrayXXf &in_img, uint64_t in_timestamp) {
  // the result
  vector<uint64_t> ts;
  vector<uint32_t> x;
  vector<uint32_t> y;
  vector<bool> pol;

  if (in_img.IsRowMajor != m_reference.IsRowMajor || 
      in_img.IsRowMajor != m_img.IsRowMajor ||
      in_img.IsRowMajor != m_C.IsRowMajor)
    throw invalid_argument("Images and threshold matrix"
       " have to have the same storage order");

  const auto rows = in_img.rows();
  const auto cols = in_img.cols();
  if (m_img.IsRowMajor) {
    for (auto row = 0U; row < rows; ++row) {
      auto cur_row = m_img.row(row).data();
      auto ref_row = m_reference.row(row).data();
      auto nxt_row = in_img.row(row).data();
      auto c_row = m_C.row(row).data();
      for (auto col = 0U; col < cols; ++col) {
        process_el(ts, x, y, pol, ref_row[col], cur_row[col], nxt_row[col],
          m_timestamp, in_timestamp, c_row[col], col, row);
      }
    }
  } else {
    for (auto col = 0U; col < cols; ++col) {
      auto cur_col = m_img.col(col).data();
      auto ref_col = m_reference.col(col).data();
      auto nxt_col = in_img.col(col).data();
      auto c_col = m_C.col(col).data();
      for (auto row = 0U; row < rows; ++row) {
        process_el(ts, x, y, pol, ref_col[row], cur_col[row], nxt_col[row],
          m_timestamp, in_timestamp, c_col[row], col, row);
      }
    }
  }

  m_timestamp = in_timestamp;
  m_img = move(in_img);
  // argsort
  vector<size_t> idx_array(ts.size());
  iota(idx_array.begin(), idx_array.end(), 0);
  sort(idx_array.begin(), idx_array.end(),
       [&ts](size_t i, size_t j) {return ts[i] < ts[j];});
  return make_tuple(rearrange(ts, idx_array), rearrange(x, idx_array),
      rearrange(y, idx_array), rearrange(pol, idx_array));
}
