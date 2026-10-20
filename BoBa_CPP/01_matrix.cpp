#include "BOBA/boba.hpp"
#include <iostream>

constexpr auto cpu = boba::execution_space::CPU;
using Matrix = boba::Matrix<cpu, double>;

int main() {
    boba::init();
    {
        Matrix A({2, 3}); // Two rows, three columns.
        A.fill_with(1.0);
        A({0, 1}) = 5.0; // First row, second column.
        A.rename("A");
        A.print();

        auto At = A.transpose();
        auto B = A * At; // Matrix multiplication.
        B.rename("B");
        B.print();

        std::cout << "B has " << B.rows() << " rows and "
                  << B.cols() << " columns.\n";
    }
    boba::finalize();
}
