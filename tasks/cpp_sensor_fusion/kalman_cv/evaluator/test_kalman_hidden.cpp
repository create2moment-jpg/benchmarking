// Hidden evaluator test. Each CHECK is independent; exceptions are caught and count as FAIL.
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>

#include "kalman_cv.hpp"

using fusion::ConstantVelocityKF;
using P4 = std::array<double, 4>;

static int failures = 0;
static void check(const char* name, const std::function<bool()>& fn) {
    bool ok = false;
    try {
        ok = fn();
    } catch (...) {
        ok = false;
    }
    std::printf("CHECK %s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}
template <class F>
static bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}
static bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol * (1.0 + std::fabs(b)); }

// Independent reference math for one predict/update step.
static P4 ref_predict(const P4& p, double q, double dt) {
    const double p00 = p[0] + dt * (p[1] + p[2]) + dt * dt * p[3];
    const double p01 = p[1] + dt * p[3];
    const double p10 = p[2] + dt * p[3];
    return {p00 + q * dt * dt * dt / 3.0, p01 + q * dt * dt / 2.0, p10 + q * dt * dt / 2.0, p[3] + q * dt};
}

#ifndef ELB_GROUP
#define ELB_GROUP 0  // 0 = all, 1 = numeric, 2 = input validation
#endif

int main() {
#if ELB_GROUP != 2
    check("predict_state", [] {
        ConstantVelocityKF kf(1.0, 2.0, {1, 0, 0, 1}, 0.5, 1.0);
        kf.predict(0.5);
        return near(kf.state().position, 2.0) && near(kf.state().velocity, 2.0);
    });
    check("predict_covariance_with_process_noise", [] {
        ConstantVelocityKF kf(0.0, 0.0, {1, 0.2, 0.2, 2}, 0.5, 1.0);
        kf.predict(1.0);
        const P4 e = ref_predict({1, 0.2, 0.2, 2}, 0.5, 1.0);
        const P4 p = kf.covariance();
        return near(p[0], e[0]) && near(p[1], e[1]) && near(p[2], e[2]) && near(p[3], e[3]);
    });
    check("update_gain_and_state", [] {
        ConstantVelocityKF kf(0.0, 1.0, {4, 1, 1, 2}, 0.0, 1.0);
        kf.update(2.0);
        const double s = 5.0, k0 = 4.0 / s, k1 = 1.0 / s;
        return near(kf.state().position, k0 * 2.0) && near(kf.state().velocity, 1.0 + k1 * 2.0);
    });
    check("update_covariance_joseph_equivalent", [] {
        ConstantVelocityKF kf(0.0, 0.0, {4, 1, 1, 2}, 0.0, 1.0);
        kf.update(0.0);
        // Exact posterior for this case: P - K S K^T
        const double s = 5.0;
        const P4 e = {4 - 16 / s, 1 - 4 / s, 1 - 4 / s, 2 - 1 / s};
        const P4 p = kf.covariance();
        return near(p[0], e[0]) && near(p[1], e[1]) && near(p[2], e[2]) && near(p[3], e[3]);
    });
    check("dt_zero_is_noop", [] {
        ConstantVelocityKF kf(3.0, -1.0, {1, 0.1, 0.1, 1}, 1.0, 1.0);
        const P4 before = kf.covariance();
        kf.predict(0.0);
        return kf.state().position == 3.0 && kf.covariance() == before;
    });
#endif
#if ELB_GROUP != 1
    check("negative_dt_throws", [] {
        ConstantVelocityKF kf(0, 0, {1, 0, 0, 1}, 1.0, 1.0);
        return throws_invalid([&] { kf.predict(-0.1); });
    });
    check("nan_measurement_throws_and_keeps_state", [] {
        ConstantVelocityKF kf(1.0, 1.0, {1, 0, 0, 1}, 1.0, 1.0);
        const bool threw = throws_invalid([&] { kf.update(std::numeric_limits<double>::quiet_NaN()); });
        return threw && kf.state().position == 1.0 && kf.covariance()[0] == 1.0;
    });
    check("ctor_rejects_nonpositive_r", [] {
        return throws_invalid([] { ConstantVelocityKF kf(0, 0, {1, 0, 0, 1}, 1.0, 0.0); });
    });
    check("ctor_rejects_negative_q", [] {
        return throws_invalid([] { ConstantVelocityKF kf(0, 0, {1, 0, 0, 1}, -1.0, 1.0); });
    });
    check("ctor_rejects_asymmetric_covariance", [] {
        return throws_invalid([] { ConstantVelocityKF kf(0, 0, {1, 0.5, 0.0, 1}, 1.0, 1.0); });
    });
#endif
#if ELB_GROUP != 2
    check("long_run_symmetric_psd", [] {
        ConstantVelocityKF kf(0.0, 0.0, {100, 0, 0, 100}, 0.05, 1e-6);
        double t = 0.0;
        for (int i = 0; i < 2000; ++i) {
            const double dt = 0.01 + 0.09 * ((i * 7919) % 100) / 100.0;
            t += dt;
            kf.predict(dt);
            kf.update(3.0 * t + ((i % 5) - 2) * 1e-4);
        }
        const P4 p = kf.covariance();
        return std::fabs(p[1] - p[2]) <= 1e-15 * (1.0 + std::fabs(p[1])) && p[0] > 0 && p[3] > 0 &&
               p[0] * p[3] - p[1] * p[2] >= -1e-18;
    });
    check("velocity_converges", [] {
        ConstantVelocityKF kf(0.0, 0.0, {10, 0, 0, 10}, 0.01, 0.01);
        for (int i = 1; i <= 100; ++i) {
            kf.predict(0.1);
            kf.update(2.0 * 0.1 * i);
        }
        return std::fabs(kf.state().velocity - 2.0) < 0.05;
    });
#endif
    return failures ? 1 : 0;
}
