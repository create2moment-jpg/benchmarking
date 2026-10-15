#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "association.hpp"

using fusion::Assignment;
using fusion::greedy_associate;

#ifndef ELB_GROUP
#define ELB_GROUP 0
#endif

static int failures = 0;
static void check(const char* name, const std::function<bool()>& fn) {
    bool ok = false;
    try { ok = fn(); } catch (...) { ok = false; }
    std::printf("CHECK %s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}
template <class F>
static bool throws_invalid(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}
static bool is(const std::vector<Assignment>& a, std::vector<std::pair<std::size_t, std::size_t>> e) {
    if (a.size() != e.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].track != e[i].first || a[i].detection != e[i].second) return false;
    return true;
}
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

int main() {
#if ELB_GROUP != 2
    check("global_not_rowwise", [] {
        // Row-wise greedy gives track0->det0 (1.0) and track1 nothing. Global greedy picks (1,0)=0.5 first.
        const std::vector<double> c = {1.0, 3.0, 0.5, 9.0};
        return is(greedy_associate(c, 2, 2, 5.0), {{0, 1}, {1, 0}});
    });
    check("no_double_assignment", [] {
        const std::vector<double> c = {1.0, 8.0, 1.5, 8.0, 2.0, 8.0};
        const auto a = greedy_associate(c, 3, 2, 5.0);
        return is(a, {{0, 0}});
    });
    check("gate_inclusive", [] {
        return is(greedy_associate({2.0}, 1, 1, 2.0), {{0, 0}});
    });
    check("tie_break_track_then_detection", [] {
        const std::vector<double> c = {1.0, 1.0, 1.0, 1.0};
        return is(greedy_associate(c, 2, 2, 5.0), {{0, 0}, {1, 1}});
    });
    check("rectangular_more_detections", [] {
        const std::vector<double> c = {4.0, 0.2, 3.0, 0.1, 5.0, 6.0};
        return is(greedy_associate(c, 2, 3, 10.0), {{0, 1}, {1, 0}});
    });
    check("costs_reported", [] {
        const auto a = greedy_associate({0.25}, 1, 1, 1.0);
        return a.size() == 1 && a[0].cost == 0.25;
    });
#endif
#if ELB_GROUP != 1
    check("inf_and_nan_infeasible", [] {
        const std::vector<double> c = {INF, NaN, NaN, 0.5};
        return is(greedy_associate(c, 2, 2, 100.0), {{1, 1}});
    });
    check("empty_inputs", [] {
        return greedy_associate({}, 0, 0, 1.0).empty() && greedy_associate({}, 0, 5, 1.0).empty() &&
               greedy_associate({}, 3, 0, 1.0).empty();
    });
    check("size_mismatch_throws", [] { return throws_invalid([] { greedy_associate({1.0, 2.0}, 2, 2, 1.0); }); });
    check("nan_gate_throws", [] { return throws_invalid([] { greedy_associate({1.0}, 1, 1, NaN); }); });
    check("negative_gate_throws", [] { return throws_invalid([] { greedy_associate({1.0}, 1, 1, -1.0); }); });
#endif
    return failures ? 1 : 0;
}
