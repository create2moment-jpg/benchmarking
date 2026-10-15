#include "association.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace fusion {

std::vector<Assignment> greedy_associate(const std::vector<double>& cost, std::size_t rows,
                                         std::size_t cols, double gate) {
    if (std::isnan(gate) || gate < 0.0) throw std::invalid_argument("gate must be >= 0");
    if (cols != 0 && rows > cost.size() / cols) throw std::invalid_argument("size mismatch");
    if (cost.size() != rows * cols) throw std::invalid_argument("cost size != rows * cols");

    std::vector<Assignment> candidates;
    candidates.reserve(cost.size());
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = cost[r * cols + c];
            if (std::isfinite(v) && v <= gate) candidates.push_back({r, c, v});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Assignment& a, const Assignment& b) {
        return std::tie(a.cost, a.track, a.detection) < std::tie(b.cost, b.track, b.detection);
    });
    std::vector<bool> track_used(rows, false);
    std::vector<bool> det_used(cols, false);
    std::vector<Assignment> out;
    for (const Assignment& a : candidates) {
        if (track_used[a.track] || det_used[a.detection]) continue;
        track_used[a.track] = true;
        det_used[a.detection] = true;
        out.push_back(a);
    }
    std::sort(out.begin(), out.end(),
              [](const Assignment& a, const Assignment& b) { return a.track < b.track; });
    return out;
}

}  // namespace fusion
