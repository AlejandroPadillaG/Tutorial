#include "support/phase_space.hpp"
#include "support/plot_svg.hpp"
#include <iostream>

int main() {
    boba::init();
    try {
        auto phase = read_phase_space("data/fd_chi_pi_fixedk_H1_t30_256x512.csv");
        auto F = phase.F;             // Our phase space is now a BoBa Matrix.
        std::cout << "F: " << F.rows() << " rows x " << F.cols() << " columns\n";

        SVD svd;
        svd(F);                      // Ask BoBa to compute the SVD.
        std::cout << "Available columns: " << svd.U.cols() << "\n";

        int rank = 46;               // TRY: 8, 16, 46.
        classroom::require(rank > 0 && rank <= svd.S.size(), "Choose an available positive rank.");
        svd.truncate(rank);

        auto L = svd.U * boba::diagonalize(svd.S);
        auto R = svd.V;
        auto F_small = L * R.transpose();

        auto error = boba::norm_difference_frobenius(F, F_small)
                   / boba::norm_frobenius(F);
        std::cout << "L: " << L.rows() << " x " << L.cols() << "\n";
        std::cout << "R: " << R.rows() << " x " << R.cols() << "\n";
        std::cout << "Approximation error: " << 100 * error << "%\n";

        // Provided C++ helpers save numbers and pictures.
        write_matrix(F_small, "results/svd/F_small.csv");
        classroom::write_heatmap_svg(F, phase.x, phase.v,
                                    "results/svd/original.svg", "Original phase space");
        classroom::write_heatmap_svg(F_small, phase.x, phase.v,
                                    "results/svd/approximation.svg", "Rank " + std::to_string(rank));
        std::cout << "Open the pictures in results/svd/.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        boba::finalize();
        return 1;
    }
    boba::finalize();
}
