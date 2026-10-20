#pragma once
#include "phase_space.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>

// Provided instructor helper. SVD, file I/O, and validation are outside timers.
namespace classroom {
namespace timing_detail {
using Clock = std::chrono::steady_clock;
struct Operation {
  const char* name;
  const char* label;
  const char* representation;
  std::function<Matrix()> compute;
};
struct Sample {
  std::size_t iterations;
  double elapsed_seconds, seconds_per_call, checksum;
};

inline const void* volatile observed_output = nullptr;
inline void observe(const Matrix& output) {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
  asm volatile("" : : "g"(output.const_data()) : "memory");
#else
  // Standard-C++ fallback: expose the address and fence compiler reordering.
  observed_output = output.const_data();
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

inline Sample run_batch(Operation& operation, std::size_t iterations) {
  double checksum = 0;
  const auto start = Clock::now();
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    Matrix output = operation.compute();
    observe(output);
    checksum += output({iteration % output.rows(),0});
  }
  const double elapsed = std::chrono::duration<double>(Clock::now()-start).count();
  require(std::isfinite(checksum), "A timing operation produced a non-finite result.");
  return {iterations,elapsed,elapsed/iterations,checksum};
}

inline Sample measured_batch(Operation& operation, std::size_t& iterations) {
  for (;;) {
    const auto sample = run_batch(operation,iterations);
    if (sample.elapsed_seconds >= .010) return sample;
    const auto multiplier = std::max<std::size_t>(
      2,static_cast<std::size_t>(std::ceil(.012/std::max(sample.elapsed_seconds,1e-9))));
    require(iterations <= 100000000/multiplier, "The timing batch exceeded its safety limit.");
    iterations *= multiplier;
  }
}

inline double quantile(std::vector<double> values, double fraction) {
  std::sort(values.begin(),values.end());
  const double position = fraction*(values.size()-1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  return values[lower] + (position-lower)*(values[upper]-values[lower]);
}

inline std::array<double,3> polynomial_checks(double dv) {
  Matrix polynomial({9,3});
  for (std::size_t i = 0; i < polynomial.rows(); ++i) {
    const double v = -.7+i*dv;
    polynomial({i,0}) = 3;
    polynomial({i,1}) = v;
    polynomial({i,2}) = v*v;
  }
  Matrix derivative = derivative_v(polynomial,dv);
  std::array<double,3> errors{};
  for (std::size_t i = 0; i < polynomial.rows(); ++i) {
    const std::array<double,3> exact{0,1,2*(-.7+i*dv)};
    for (std::size_t j = 0; j < 3; ++j)
      errors[j] = std::max(errors[j],std::abs(derivative({i,j})-exact[j]));
  }
  for (double error : errors)
    require(error < 2e-12, "The constant/linear/quadratic derivative check failed.");
  return errors;
}

inline std::string compiler_name() {
#if defined(__clang__)
  return "Clang " + std::string(__clang_version__);
#elif defined(__GNUC__)
  return "GCC " + std::string(__VERSION__);
#elif defined(_MSC_VER)
  return "MSVC " + std::to_string(_MSC_VER);
#else
  return "Unknown C++ compiler";
#endif
}

inline std::string json_string(const std::string& value) {
  std::string escaped = "\"";
  for (char character : value) {
    if (character == '"' || character == '\\') escaped += '\\';
    if (character == '\n') escaped += "\\n";
    else if (character == '\r') escaped += "\\r";
    else if (character == '\t') escaped += "\\t";
    else escaped += character;
  }
  return escaped+'"';
}
} // namespace timing_detail

/** Time three derivative operations on already-prepared F, L, and Rt.
 *  The factor-only result is DL with Rt reused. It is not a full matrix.
 *  All paths allocate outputs; the two full-output paths produce dense matrices.
 */
inline void compare_derivatives(const Matrix& F, const Matrix& L, const Matrix& Rt,
                                double dv, const std::filesystem::path& outdir) {
  require(L.rows() == F.rows() && Rt.cols() == F.cols() && L.cols() == Rt.rows() &&
          L.cols() > 0, "The factor shapes must satisfy F_r = L*Rt.");
  Eigen::setNbThreads(1);
  const auto polynomial_errors = timing_detail::polynomial_checks(dv);
  Matrix dense_derivative = derivative_v(F,dv);
  Matrix approximation = L*Rt;
  Matrix derivative_of_approximation = derivative_v(approximation,dv);
  Matrix derivative_from_factors = derivative_v(L,dv)*Rt;
  const double matrix_error = relative_error(approximation,F);
  const double derivative_error = relative_error(derivative_from_factors,dense_derivative);
  const double identity_error = relative_error(derivative_from_factors,derivative_of_approximation);
  require(identity_error < 1e-10, "D(L*Rt) disagrees with (D L)*Rt.");

  std::array<timing_detail::Operation,3> operations{{
    {"dense_derivative", "Full matrix derivative", "full matrix",
      [&]() { return derivative_v(F,dv); }},
    {"factor_derivative", "Differentiate L only", "DL only; Rt reused",
      [&]() { return derivative_v(L,dv); }},
    {"factor_derivative_and_reconstruct", "Factors + full reconstruction", "full matrix",
      [&]() { return derivative_v(L,dv)*Rt; }}
  }};
  for (auto& operation : operations)
    for (int warmup = 0; warmup < 5; ++warmup) timing_detail::observe(operation.compute());

  std::filesystem::create_directories(outdir);
  std::ofstream raw(outdir/"timing_samples.csv");
  require(bool(raw), "Cannot create timing_samples.csv.");
  raw.imbue(std::locale::classic());
  raw << std::setprecision(17)
      << "operation,trial,order,iterations,batch_seconds,microseconds_per_call,checksum\n";
  std::array<std::vector<double>,3> measurements;
  std::array<std::size_t,3> iterations{1,1,1}, order{0,1,2};
  std::mt19937 generator(20261020);
  for (int trial = 1; trial <= 7; ++trial) {
    std::shuffle(order.begin(),order.end(),generator);
    for (std::size_t position = 0; position < order.size(); ++position) {
      const auto method = order[position];
      const auto sample = timing_detail::measured_batch(operations[method],iterations[method]);
      measurements[method].push_back(sample.seconds_per_call);
      raw << operations[method].name << ',' << trial << ',' << position << ','
          << sample.iterations << ',' << sample.elapsed_seconds << ','
          << sample.seconds_per_call*1e6 << ',' << sample.checksum << '\n';
    }
  }

  std::array<double,3> medians{};
  for (std::size_t method = 0; method < operations.size(); ++method)
    medians[method] = timing_detail::quantile(measurements[method],.5);
  std::ofstream summary(outdir/"timings.csv");
  require(bool(summary), "Cannot create timings.csv.");
  summary.imbue(std::locale::classic());
  summary << std::setprecision(17)
          << "operation,output_representation,median_us,q25_us,q75_us,speedup_vs_dense,"
             "reconstructed_derivative_relative_error\n";
  const auto old_flags = std::cout.flags();
  const auto old_precision = std::cout.precision();
  std::cout << "\nOperation                       Output                  Median (us)  Speed vs full\n";
  for (std::size_t method = 0; method < operations.size(); ++method) {
    const double speedup = medians[0]/medians[method];
    summary << operations[method].name << ',' << operations[method].representation << ','
            << medians[method]*1e6 << ','
            << timing_detail::quantile(measurements[method],.25)*1e6 << ','
            << timing_detail::quantile(measurements[method],.75)*1e6 << ',' << speedup << ','
            << (method == 0 ? 0.0 : derivative_error) << '\n';
    std::cout << std::left << std::setw(32) << operations[method].label
              << std::setw(24) << operations[method].representation << std::right
              << std::fixed << std::setprecision(3) << std::setw(11) << medians[method]*1e6
              << std::setw(14) << speedup << "x\n";
  }
  std::cout.flags(old_flags);
  std::cout.precision(old_precision);
  std::cout << "L-only output is compressed: keep DL and reuse Rt.\n"
            << "Relative matrix error: " << matrix_error
            << "; relative discrete derivative error: " << derivative_error << ".\n"
            << "SVD, compilation, correctness checks and file I/O are outside the timers.\n";

  std::ofstream metadata(outdir/"timing_metadata.json");
  require(bool(metadata), "Cannot create timing_metadata.json.");
  metadata.imbue(std::locale::classic());
  metadata << std::setprecision(17)
           << "{\n\"rows_v\":" << F.rows() << ",\"columns_x\":" << F.cols()
           << ",\"source_data_rows\":" << F.rows()*F.cols()
           << ",\"rank\":" << L.cols() << ",\"dv\":" << dv
           << ",\"matrix_relative_error\":" << matrix_error
           << ",\"derivative_relative_error\":" << derivative_error
           << ",\"same_operator_relative_error\":" << identity_error
           << ",\"polynomial_max_abs_errors\":[" << polynomial_errors[0] << ','
           << polynomial_errors[1] << ',' << polynomial_errors[2] << ']'
           << ",\"trials\":7,\"warmup_calls\":5,\"minimum_recorded_batch_seconds\":0.010"
           << ",\"eigen_threads\":" << Eigen::nbThreads()
           << ",\"compiler\":" << timing_detail::json_string(timing_detail::compiler_name())
           << ",\"clock\":\"steady_clock wall time; repeated hot-cache operations\""
           << ",\"method_order\":\"Fixed-seed random permutation in every trial\""
           << ",\"allocation_policy\":\"Every operation returns a newly allocated output. Factor-only returns DL and reuses Rt. Reconstruction allocates a final full matrix; backend temporaries may occur.\""
           << ",\"excluded\":\"Preceding SVD, factor preparation, correctness checks, compilation, file I/O\""
           << ",\"source_layout\":\"Complete x,v,f Cartesian grid; source_data_rows excludes the header; F has ascending v rows and ascending x columns\""
           << ",\"operator\":\"Centered second-order interior; one-sided second-order endpoints; no periodic wrap\""
           << ",\"interpretation\":\"Errors compare the chosen discrete derivative, not verified continuum derivatives. Factor-only and full-matrix outputs have different representations. Repeated identical derivatives can be cached; these are operation timings, not an application speedup.\""
           << ",\"all_checks_passed\":true\n}\n";
  raw.close();
  summary.close();
  metadata.close();
  require(bool(raw) && bool(summary) && bool(metadata), "Could not finish writing timing results.");
}
} // namespace classroom

using classroom::compare_derivatives;
