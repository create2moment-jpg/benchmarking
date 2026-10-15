#include "association.hpp"

namespace fusion {

std::vector<Assignment> greedy_associate(const std::vector<double>& cost, std::size_t rows,
                                         std::size_t cols, double gate) {
    std::vector<Assignment> out;
    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t best = cols;
        for (std::size_t c = 0; c < cols; ++c) {
            if (cost[r * cols + c] < gate && (best == cols || cost[r * cols + c] < cost[r * cols + best])) {
                best = c;
            }
        }
        if (best != cols) out.push_back({r, best, cost[r * cols + best]});
    }
    return out;
}

}  // namespace fusion
