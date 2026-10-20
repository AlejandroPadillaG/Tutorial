#pragma once
#include "BOBA/boba.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Provided classroom helpers: CSV handling and the chosen derivative stencil.
namespace classroom {
inline constexpr auto cpu = boba::execution_space::CPU;
using Matrix = boba::Matrix<cpu, double>;
using SVD = boba::SVD<cpu, double>;

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

struct PhaseSpace {
  Matrix F;                 // F({i,j}) = f(v[i], x[j]).
  std::vector<double> x, v;  // Both coordinates are sorted in ascending order.
  double dv;
};

namespace csv_detail {
inline std::string trim(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last-first+1);
}

inline std::array<std::string,3> fields(const std::string& line,
                                      std::size_t line_number) {
  const auto first = line.find(',');
  const auto second = first == std::string::npos ? first : line.find(',',first+1);
  require(first != std::string::npos && second != std::string::npos &&
          line.find(',',second+1) == std::string::npos,
          "Expected exactly three CSV fields on line " + std::to_string(line_number));
  return {trim(line.substr(0,first)), trim(line.substr(first+1,second-first-1)),
          trim(line.substr(second+1))};
}

inline double number(const std::string& text, std::size_t line_number) {
  std::istringstream input(text);
  input.imbue(std::locale::classic());
  double value;
  require(bool(input >> value) && std::isfinite(value),
          "Expected a finite number on CSV line " + std::to_string(line_number));
  input >> std::ws;
  require(input.eof(), "Unexpected text on CSV line " + std::to_string(line_number));
  return value;
}

inline void sort_unique(std::vector<double>& values) {
  std::sort(values.begin(),values.end());
  values.erase(std::unique(values.begin(),values.end()),values.end());
}
} // namespace csv_detail

/** Read an unquoted numeric x,v,f CSV and preserve every sample.
 *  Reject duplicates, incomplete Cartesian grids, and nonuniform velocity grids.
 */
inline PhaseSpace read_phase_space(const std::filesystem::path& path) {
  std::ifstream input(path);
  require(bool(input), "Cannot open " + path.string());
  std::string line;
  require(bool(std::getline(input,line)), "The input CSV is empty.");
  if (line.compare(0,3,"\xEF\xBB\xBF") == 0) line.erase(0,3); // Optional UTF-8 BOM.
  require(csv_detail::fields(line,1) == std::array<std::string,3>{"x","v","f"},
          "The CSV header must be x,v,f.");

  struct Record { double x, v, f; };
  std::vector<Record> records;
  std::vector<double> x, v;
  std::size_t line_number = 1;
  while (std::getline(input,line)) {
    ++line_number;
    if (csv_detail::trim(line).empty()) continue;
    const auto fields = csv_detail::fields(line,line_number);
    Record record{csv_detail::number(fields[0],line_number),
                  csv_detail::number(fields[1],line_number),
                  csv_detail::number(fields[2],line_number)};
    records.push_back(record);
    x.push_back(record.x);
    v.push_back(record.v);
  }
  require(input.eof(), "Could not finish reading " + path.string());
  require(!records.empty(), "The CSV has no data rows.");
  csv_detail::sort_unique(x);
  csv_detail::sort_unique(v);
  require(v.size() >= 3, "At least three velocity coordinates are needed.");
  require(v.size() <= std::numeric_limits<std::size_t>::max()/x.size(),
          "The requested Cartesian grid is too large.");
  require(records.size() == x.size()*v.size(),
          "The CSV does not contain exactly one sample per Cartesian grid point.");

  const double dv = (v.back()-v.front())/double(v.size()-1);
  require(dv > 0 && std::isfinite(dv), "The velocity spacing must be finite and positive.");
  const double coordinate_scale = std::max({1.0,std::abs(v.front()),std::abs(v.back())});
  const double spacing_tolerance = 1e-10*dv +
    64*std::numeric_limits<double>::epsilon()*coordinate_scale;
  for (std::size_t i = 1; i < v.size(); ++i)
    require(std::abs((v[i]-v[i-1])-dv) <= spacing_tolerance,
            "The velocity coordinates must be uniformly spaced.");

  Matrix F({v.size(),x.size()});
  std::vector<bool> seen(records.size(),false);
  for (const auto& record : records) {
    const auto i = std::size_t(std::lower_bound(v.begin(),v.end(),record.v)-v.begin());
    const auto j = std::size_t(std::lower_bound(x.begin(),x.end(),record.x)-x.begin());
    const auto index = i*x.size()+j;
    require(!seen[index], "The CSV contains a duplicate (x,v) sample.");
    seen[index] = true;
    F({i,j}) = record.f;
  }
  require(std::all_of(seen.begin(),seen.end(),[](bool present) { return present; }),
          "The CSV is missing a Cartesian grid sample.");
  return {std::move(F),std::move(x),std::move(v),dv};
}

/** Write a numeric matrix CSV to the explicit output path. */
inline void write_matrix(const Matrix& matrix, const std::filesystem::path& path) {
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  require(bool(output), "Cannot create " + path.string());
  output.imbue(std::locale::classic());
  output << std::setprecision(17);
  for (std::size_t i = 0; i < matrix.rows(); ++i) {
    for (std::size_t j = 0; j < matrix.cols(); ++j) {
      if (j > 0) output << ',';
      output << matrix({i,j});
    }
    output << '\n';
  }
  output.close();
  require(bool(output), "Could not finish writing " + path.string());
}

/** Relative Frobenius error; the reference matrix must be nonzero. */
inline double relative_error(const Matrix& approximation, const Matrix& reference) {
  require(approximation.rows() == reference.rows() &&
          approximation.cols() == reference.cols(), "The compared matrix shapes must agree.");
  const double reference_norm = boba::norm_frobenius(reference);
  require(reference_norm > 0, "Relative error needs a nonzero reference matrix.");
  return boba::norm_difference_frobenius(approximation,reference)/reference_norm;
}

/** Differentiate along the rows. Each call allocates a new output.
 *  Second-order centered interior and one-sided endpoints; no periodic wrap.
 */
inline Matrix derivative_v(const Matrix& A, double dv) {
  require(A.rows() >= 3 && A.cols() > 0 && dv > 0 && std::isfinite(dv),
          "Use at least three velocity rows, one column, and positive finite dv.");
  Matrix derivative({A.rows(),A.cols()});
  for (std::size_t j = 0; j < A.cols(); ++j) {
    derivative({0,j}) = (-3*A({0,j}) + 4*A({1,j}) - A({2,j})) / (2*dv);
    for (std::size_t i = 1; i+1 < A.rows(); ++i)
      derivative({i,j}) = (A({i+1,j}) - A({i-1,j})) / (2*dv);
    const std::size_t last = A.rows()-1;
    derivative({last,j}) = (3*A({last,j}) - 4*A({last-1,j}) + A({last-2,j})) / (2*dv);
  }
  return derivative;
}
} // namespace classroom

using classroom::cpu;
using classroom::Matrix;
using classroom::SVD;
using classroom::PhaseSpace;
using classroom::read_phase_space;
using classroom::write_matrix;
using classroom::relative_error;
using classroom::derivative_v;
