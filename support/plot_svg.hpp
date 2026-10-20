#pragma once
#include "phase_space.hpp"
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
#include <vector>

// Optional plotting helper: writes a browser-readable SVG using only C++.
namespace classroom {
namespace svg_detail {

inline std::string escape_xml(const std::string& text) {
  std::string result;
  for (const char c : text) {
    switch (c) {
      case '&': result += "&amp;"; break;
      case '<': result += "&lt;"; break;
      case '>': result += "&gt;"; break;
      case '"': result += "&quot;"; break;
      case '\'': result += "&apos;"; break;
      default: result += c;
    }
  }
  return result;
}

inline double uniform_spacing(const std::vector<double>& grid,
                              const std::string& name) {
  if (grid.size() < 2)
    throw std::invalid_argument(name + " needs at least two cell centers.");
  for (double value : grid)
    if (!std::isfinite(value))
      throw std::invalid_argument(name + " contains a nonfinite coordinate.");
  const double step = grid[1] - grid[0];
  if (!(step > 0) || !std::isfinite(step))
    throw std::invalid_argument(name + " must increase strictly.");
  for (std::size_t i = 1; i < grid.size(); ++i) {
    const double difference = grid[i] - grid[i-1];
    const double roundoff = 64 * std::numeric_limits<double>::epsilon() *
                           std::max(std::abs(grid[i]), std::abs(grid[i-1]));
    if (!(difference > 0) || !std::isfinite(difference) ||
        std::abs(difference-step) > 1e-8*step + roundoff)
      throw std::invalid_argument(name + " must be a uniform increasing grid.");
  }
  return step;
}

inline std::string label(double value) {
  std::ostringstream text;
  text.imbue(std::locale::classic());
  text << std::setprecision(4) << value;
  return text.str();
}

// 0 = blue, 128 = white, 256 = red. Negative values stay negative.
inline std::string color(int bin) {
  const std::array<int,3> endpoint =
      bin < 128 ? std::array<int,3>{42,86,163}
                : std::array<int,3>{192,55,56};
  const double strength = std::abs(bin-128) / 128.0;
  std::ostringstream rgb;
  rgb << "rgb(";
  for (int channel = 0; channel < 3; ++channel) {
    if (channel) rgb << ',';
    rgb << std::lround(255 + strength*(endpoint[channel]-255));
  }
  return rgb.str() + ')';
}
} // namespace svg_detail

/** Write every original matrix cell; rows are v and columns are x.
 *  x and v are finite, increasing, uniform cell-center coordinates.
 *  color_limit > 0 fixes the symmetric scale [-color_limit,+color_limit].
 *  color_limit == 0 uses max(abs(A)); an all-zero matrix uses [-1,+1].
 *  Cells outside a fixed color scale saturate at the corresponding endpoint.
 */
inline void write_heatmap_svg(const Matrix& A, const std::vector<double>& x,
                             const std::vector<double>& v,
                             const std::filesystem::path& path,
                             const std::string& title,
                             double color_limit = 0) {
  if (A.rows() != v.size() || A.cols() != x.size())
    throw std::invalid_argument("Heatmap shape must be rows=v, columns=x.");
  if (!std::isfinite(color_limit) || color_limit < 0)
    throw std::invalid_argument("color_limit must be positive, or zero for auto.");
  const double dx = svg_detail::uniform_spacing(x, "x");
  const double dv = svg_detail::uniform_spacing(v, "v");
  const double xmin = x.front()-dx/2, xmax = x.back()+dx/2;
  const double vmin = v.front()-dv/2, vmax = v.back()+dv/2;
  const double xspan = xmax-xmin, vspan = vmax-vmin;
  if (!std::isfinite(xmin) || !std::isfinite(xmax) ||
      !std::isfinite(vmin) || !std::isfinite(vmax) ||
      !std::isfinite(xspan) || !std::isfinite(vspan) ||
      !(xspan > 0) || !(vspan > 0))
    throw std::invalid_argument("Heatmap coordinate bounds are not representable.");
  constexpr double left = 110, top = 100, width = 740, height = 520;
  const double xscale = width/xspan, vscale = height/vspan;
  if (!std::isfinite(xscale) || !std::isfinite(vscale))
    throw std::invalid_argument("Heatmap coordinates are too tightly spaced.");

  double maximum = 0;
  for (std::size_t i = 0; i < A.rows(); ++i)
    for (std::size_t j = 0; j < A.cols(); ++j) {
      const double value = A({i,j});
      if (!std::isfinite(value))
        throw std::invalid_argument("Heatmap contains a nonfinite matrix value.");
      maximum = std::max(maximum, std::abs(value));
    }
  const bool zero_auto_scale = color_limit == 0 && maximum == 0;
  if (color_limit == 0) color_limit = maximum == 0 ? 1 : maximum;

  // One path per color keeps the SVG small without dropping or averaging cells.
  std::array<std::ostringstream,257> cells;
  for (auto& cell_path : cells) {
    cell_path.imbue(std::locale::classic());
    cell_path << std::setprecision(17);
  }
  for (std::size_t i = 0; i < A.rows(); ++i)
    for (std::size_t j = 0; j < A.cols(); ++j) {
      const double value = A({i,j});
      const double scaled = value <= -color_limit ? -1 :
                            value >= color_limit ? 1 : value/color_limit;
      const int bin = static_cast<int>(std::lround(128*(scaled+1)));
      cells[bin] << 'M' << x[j]-dx/2 << ',' << v[i]-dv/2
                 << 'h' << dx << 'v' << dv << 'h' << -dx << 'z';
    }

  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot create SVG: " + path.string());
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  const auto tick_label = [&](double px, double py, const std::string& text,
                              const char* anchor) {
    out << "<text x=\"" << px << "\" y=\"" << py << "\" text-anchor=\""
        << anchor << "\">" << text << "</text>\n";
  };
  const std::string safe_title = svg_detail::escape_xml(title);
  out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
      << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1100\" height=\"760\" "
         "viewBox=\"0 0 1100 760\" role=\"img\" aria-labelledby=\"title description\">\n"
      << "<title id=\"title\">" << safe_title << "</title>\n"
      << "<desc id=\"description\">Signed phase-space heatmap. Position x is "
         "horizontal; velocity v increases upward. Every original matrix cell "
         "is included. The color scale is symmetric about zero.</desc>\n"
      << "<defs><linearGradient id=\"scale\" x1=\"0\" y1=\"1\" x2=\"0\" y2=\"0\">"
         "<stop offset=\"0\" stop-color=\"rgb(42,86,163)\"/>"
         "<stop offset=\"0.5\" stop-color=\"white\"/>"
         "<stop offset=\"1\" stop-color=\"rgb(192,55,56)\"/>"
         "</linearGradient><clipPath id=\"plot-area\"><rect x=\"110\" y=\"100\" "
         "width=\"740\" height=\"520\"/></clipPath></defs>\n"
      << "<rect width=\"1100\" height=\"760\" fill=\"white\"/>\n"
      << "<g font-family=\"Arial,Helvetica,sans-serif\" fill=\"#18314a\">\n"
      << "<text x=\"550\" y=\"42\" text-anchor=\"middle\" font-size=\"23\" "
         "font-weight=\"bold\">" << safe_title << "</text>\n"
      << "<g clip-path=\"url(#plot-area)\"><g shape-rendering=\"crispEdges\" "
         "transform=\"translate(" << left << ',' << top+height << ") scale("
      << xscale << ',' << -vscale << ") translate("
      << -xmin << ',' << -vmin << ")\">\n";
  for (int bin = 0; bin <= 256; ++bin) {
    const std::string geometry = cells[bin].str();
    if (!geometry.empty())
      out << "<path fill=\"" << svg_detail::color(bin)
          << "\" d=\"" << geometry << "\"/>\n";
  }
  out << "</g></g><rect x=\"110\" y=\"100\" width=\"740\" height=\"520\" "
         "fill=\"none\" stroke=\"#18314a\" stroke-width=\"1.2\"/>\n"
      << "<g font-size=\"17\">\n";
  for (int k = 0; k <= 4; ++k) {
    const double fraction = k/4.0;
    const double px = left+fraction*width, py = top+(1-fraction)*height;
    out << "<path d=\"M" << px << ',' << top+height << "v7M" << left << ','
        << py << "h-7\" fill=\"none\" stroke=\"#18314a\"/>\n";
    tick_label(px, top+height+30, svg_detail::label(xmin+fraction*xspan), "middle");
    tick_label(left-14, py+6, svg_detail::label(vmin+fraction*vspan), "end");
  }
  out << "</g><text x=\"480\" y=\"690\" text-anchor=\"middle\" font-size=\"21\">"
         "Position x</text><text transform=\"translate(30,360) rotate(-90)\" "
         "text-anchor=\"middle\" font-size=\"21\">Velocity v</text>\n"
      << "<rect x=\"914\" y=\"100\" width=\"26\" height=\"520\" "
         "fill=\"url(#scale)\" stroke=\"#18314a\"/>\n"
      << "<g font-size=\"16\">\n";
  tick_label(927, 80, "Signed value", "middle");
  for (int k = 0; k <= 4; ++k) {
    const double fraction = k/4.0;
    tick_label(952, top+(1-fraction)*height+5,
               svg_detail::label((2*fraction-1)*color_limit), "start");
  }
  out << "</g><text x=\"550\" y=\"730\" text-anchor=\"middle\" font-size=\"14\">"
      << A.rows() << " velocity rows × " << A.cols()
      << " position columns; all cells shown; 257-color signed scale.";
  if (zero_auto_scale) out << " All values are zero; display limits are ±1.";
  else if (maximum > color_limit) out << " Colors saturate at the stated limits.";
  out << "</text></g></svg>\n";
  out.close();
  if (!out) throw std::runtime_error("Failed while writing SVG: " + path.string());
}
} // namespace classroom
