#include "support/phase_space.hpp"
#include "support/plot_svg.hpp"
#include <iostream>

int main() {
    boba::init();
    try {
        auto phase = read_phase_space("data/fd_chi_pi_fixedk_H1_t30_256x512.csv");
        auto F = phase.F;
        int rank = 46;               // TRY: use the same rank as in lesson 2.

        SVD svd;
        svd(F);
        classroom::require(rank > 0 && rank <= svd.S.size(), "Choose an available positive rank.");
        svd.truncate(rank);
        auto L = svd.U * boba::diagonalize(svd.S);
        auto R = svd.V;

        // derivative_v is a supplied C++ finite-difference function.
        auto dF = derivative_v(F, phase.dv);      // Differentiate the full matrix.
        auto dL = derivative_v(L, phase.dv);      // Differentiate the smaller matrix.
        auto dF_small = dL * R.transpose();      // Reconstruct with BoBa.

        auto error = boba::norm_difference_frobenius(dF, dF_small)
                   / boba::norm_frobenius(dF);
        std::cout << "Derivative error: " << 100 * error << "%\n";
        std::cout << "We differentiated " << F.cols() << " columns in F, but only "
                  << L.cols() << " columns in L.\n";

        write_matrix(dF, "results/derivative/full.csv");
        write_matrix(dF_small, "results/derivative/low_rank.csv");
        classroom::write_heatmap_svg(dF, phase.x, phase.v,
                                    "results/derivative/full.svg", "Derivative of F");
        classroom::write_heatmap_svg(dF_small, phase.x, phase.v,
                                    "results/derivative/low_rank.svg", "Derivative from L and R");
        std::cout << "Open the pictures in results/derivative/.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        boba::finalize();
        return 1;
    }
    boba::finalize();
}
