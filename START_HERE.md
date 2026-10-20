# BoBa in C++: edit, compile, run

Open this folder in **Visual Studio Code**. Work through the four `.cpp` files.
After editing a file, compile it and run it again in **Terminal → New Terminal**.
All calculations, CSV loading, pictures, and timings are produced by C++.

## What students need

1. **Visual Studio Code**, with Microsoft's **C/C++** extension.
2. A **C++20 compiler**: Clang on macOS, or a recent GCC/Clang on Linux.
3. The **BoBa source archive provided by the instructor**, extracted into a folder.
4. **Eigen headers**, used by BoBa's CPU linear algebra backend. This tutorial uses
   Eigen 5.0.1: download its source archive from the
   [official tagged source page](https://gitlab.com/libeigen/eigen/-/tree/5.0.1).
5. This tutorial folder. Your original phase-space CSV is already in `data/`.

Eigen is a header-only dependency: download and extract its source, then point the
compiler at the folder containing `Eigen/Core`. There is no separate Eigen library
to compile. [Eigen setup instructions](https://libeigen.gitlab.io/eigen/docs-3.4/GettingStarted.html)

If you need a compiler:

| Computer | Setup |
|---|---|
| macOS | Run `xcode-select --install` in Terminal. |
| Ubuntu Linux | Run `sudo apt update`, then `sudo apt install build-essential`. Use a distribution with a C++20 compiler, such as Ubuntu 24.04. |
| Windows | Use Ubuntu in WSL, open this folder through the VS Code WSL extension, and follow the Ubuntu commands inside that terminal. |

The Windows route uses a Linux compiler through WSL. Native MSVC is not tested for
this tutorial. See Microsoft's [macOS compiler guide](https://code.visualstudio.com/docs/cpp/config-clang-mac),
[Linux compiler guide](https://code.visualstudio.com/docs/cpp/config-linux), and
[VS Code with WSL guide](https://code.visualstudio.com/docs/cpp/config-wsl).

## First run

Keep the three extracted folders together, for example:

```text
course/
  BoBa-main-2/
    include/BOBA/boba.hpp
    source/boba.cpp
  eigen-5.0.1/
    Eigen/Core
  BoBa_CPP_Classroom/
    01_matrix.cpp
    data/
    support/
```

Open **BoBa_CPP_Classroom** in VS Code and open its terminal.
Set these two paths once per terminal session; change the names if yours differ:

```bash
BOBA="../BoBa-main-2"
EIGEN="../eigen-5.0.1"
```

If Eigen is already installed, use its include folder instead. Common examples:
`EIGEN="/usr/include/eigen3"` on Ubuntu or
`EIGEN="/opt/homebrew/include/eigen3"` on Apple Silicon Macs.

Now compile the first example:

```bash
c++ -std=c++20 -O3 -DBOBA_CPU -DBOBA_ENABLE_EIGEN \
    -I"$BOBA/include" -I"$EIGEN" \
    01_matrix.cpp "$BOBA/source/boba.cpp" -o lesson
```

When compilation finishes successfully, run it:

```bash
./lesson
```

`c++` invokes the compiler. `-I` tells it where to find headers. The two `-D` flags
select BoBa's CPU/Eigen backend. `source/boba.cpp` supplies BoBa's initialization
code. `-o lesson` names the executable.

Use this terminal command to build the whole example with BoBa's source file.
The editor's generic “run active file” action does not know these extra flags.

## 1. Create, change, transpose, multiply

Open `01_matrix.cpp`. The lines to experiment with are:

```cpp
Matrix A({2, 3});
A.fill_with(1.0);
A({0, 1}) = 5.0;
A.print();

auto At = A.transpose();
auto B = A * At;
B.print();
```

`Matrix` is a short name for a BoBa CPU matrix of `double` values.
`auto` lets C++ infer the result's type. `*` performs matrix multiplication.

**Try:** change 5 to 9, save, compile again, and run again.
Then replace `A * At` with `At * A`. Predict the result's shape first.

In C++ code, indices start at zero. BoBa's `print()` labels entries starting at one:
`A({0,1})` appears as `A(1,2)` in its printout.

The surrounding `main()` is the program's entry point. Keep `boba::init()` and
`boba::finalize()`. The inner braces let matrices be destroyed before finalization.

## 2. Compute an SVD of your phase space

Open `02_phase_space_svd.cpp`. Compile the same way, changing the source filename:

```bash
c++ -std=c++20 -O3 -DBOBA_CPU -DBOBA_ENABLE_EIGEN \
    -I"$BOBA/include" -I"$EIGEN" \
    02_phase_space_svd.cpp "$BOBA/source/boba.cpp" -o lesson
./lesson
```

This program reads **fd_chi_pi_fixedk_H1_t30_256x512.csv** directly in C++.
Its matrix F has **512 velocity rows and 256 position columns**.
The supplied loader preserves signed values and checks the coordinate grid.

Focus on these BoBa commands:

```cpp
SVD svd;
svd(F);
svd.truncate(rank);

auto L = svd.U * boba::diagonalize(svd.S);
auto R = svd.V;
auto F_small = L * R.transpose();
```

`SVD` is a short name for BoBa's SVD tool. `truncate(rank)` keeps that many columns.
`diagonalize` turns the singular values into a diagonal matrix. BoBa stores V,
so the reconstruction uses `R.transpose()`.

**Try:** change `rank = 46` to `rank = 8`. Save, compile, and run again.
Read the error printed in the terminal and open the SVG pictures in `results/svd/`
in a browser or image preview. The pictures use x horizontally and v vertically.

Only choose positive ranks no larger than the available SVD columns printed by
the program. BoBa can discard numerically tiny singular values.

## 3. Differentiate in velocity

Compile and run `03_velocity_derivative.cpp`:

```bash
c++ -std=c++20 -O3 -DBOBA_CPU -DBOBA_ENABLE_EIGEN \
    -I"$BOBA/include" -I"$EIGEN" \
    03_velocity_derivative.cpp "$BOBA/source/boba.cpp" -o lesson
./lesson
```

The new operations are:

```cpp
auto dF = derivative_v(F, phase.dv);
auto dL = derivative_v(L, phase.dv);
auto dF_small = dL * R.transpose();
```

`derivative_v` is a supplied C++ function built with BoBa matrix entry access.
It is not a built-in BoBa method. Inside the grid, its central line is:

```cpp
derivative({i,j}) = (A({i+1,j}) - A({i-1,j})) / (2*dv);
```

The function also handles the two end rows with one-sided differences.
You can read its complete implementation in `support/phase_space.hpp`.

**Try:** change the rank and compare the derivative error and pictures.
This example is a separate program; set its rank explicitly.
At rank 46, the supplied data gives about **0.692% derivative error**.
The error compares two applications of the same discrete formula, not an exact
physical derivative.

## 4. Check whether it is faster

Compile and run `04_compare_speed.cpp` with optimization enabled:

```bash
c++ -std=c++20 -O3 -DBOBA_CPU -DBOBA_ENABLE_EIGEN \
    -I"$BOBA/include" -I"$EIGEN" \
    04_compare_speed.cpp "$BOBA/source/boba.cpp" -o lesson
./lesson
```

The supplied timing helper compares three C++ expressions:

| Expression | What it produces |
|---|---|
| `derivative_v(F, dv)` | A full derivative matrix |
| `derivative_v(L, dv)` | The derivative factor, reusing R |
| `derivative_v(L, dv) * Rt` | A reconstructed full derivative matrix |

Read the median times printed in the terminal. Keeping factors and reconstructing
a full matrix answer different questions. The SVD, compilation, CSV reading, and
factor preparation are excluded from these timings. The initial SVD still costs
time if you start from a dense matrix.

**Try:** run ranks 8, 16, and 46. Record derivative error and the three timings.
Explain the result using the number of columns in F and L.

## Supplied helpers and common fixes

- `support/phase_space.hpp`: original CSV loader, CSV writer, finite differences.
- `support/plot_svg.hpp`: standalone SVG pictures made in C++.
- `support/benchmark.hpp`: seven timing trials, numerical checks, CSV results.

The BoBa commands being taught are in the four `.cpp` files. These helpers keep
file handling and timing machinery out of the first exercises.

| Message or symptom | Fix |
|---|---|
| `c++: command not found` | Install your platform's compiler above. |
| `BOBA/boba.hpp` not found | Check that `$BOBA/include/BOBA/boba.hpp` exists. |
| `Eigen/Core` not found | `$EIGEN` must be the folder containing the `Eigen` directory. |
| Cannot open the CSV | Run `./lesson` from the tutorial folder, where `data/` exists. |
| Old output after editing | Save the `.cpp`, compile successfully, then run `./lesson`. |
| Multiple definitions of `main` | Compile one lesson at a time, as in the commands above. |

The examples are independent programs. Run each compile command successfully
before running `./lesson`; a failed compilation can leave an older executable.
