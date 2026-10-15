// Gated greedy data association between tracks (rows) and detections (columns).
#pragma once

#include <cstddef>
#include <vector>

namespace fusion {

struct Assignment {
    std::size_t track;
    std::size_t detection;
    double cost;
};

// cost: row-major matrix of size rows * cols. Entries may be +inf or NaN (both infeasible).
// A pair is feasible when its cost is finite and cost <= gate (inclusive).
// Global greedy: repeatedly take the cheapest remaining feasible pair; ties broken by lower
// track index, then lower detection index. Each track/detection is used at most once.
// Returns assignments sorted by track index.
// Throws std::invalid_argument if cost.size() != rows * cols, or gate is NaN or negative.
std::vector<Assignment> greedy_associate(const std::vector<double>& cost, std::size_t rows,
                                         std::size_t cols, double gate);

}  // namespace fusion
