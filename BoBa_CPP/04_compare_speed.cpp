#include "support/phase_space.hpp"
#include "support/benchmark.hpp"
#include <iostream>

int main() {
    boba::init();
    try {
        auto phase = read_phase_space("data/fd_chi_pi_fixedk_H1_t30_256x512.csv");
        auto F = phase.F;
        int rank = 46;               // TRY: 8, 16, 46.

        SVD svd;
        svd(F);
        classroom::require(rank > 0 && rank <= svd.S.size(), "Choose an available positive rank.");
        svd.truncate(rank);
        auto L = svd.U * boba::diagonalize(svd.S);
        auto R = svd.V;
        auto Rt = R.transpose();

        // This C++ helper repeats and times the following three operations:
        // 1. derivative_v(F, phase.dv)          -> a full matrix
        // 2. derivative_v(L, phase.dv)          -> factors, reusing R
        // 3. derivative_v(L, phase.dv) * Rt     -> a full matrix
        classroom::compare_derivatives(F, L, Rt, phase.dv, "results/timing");
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        boba::finalize();
        return 1;
    }
    boba::finalize();
}
