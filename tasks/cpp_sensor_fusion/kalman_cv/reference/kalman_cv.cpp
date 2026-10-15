#include "kalman_cv.hpp"

#include <cmath>
#include <stdexcept>

namespace fusion {

namespace {

bool finite(double v) { return std::isfinite(v); }

}  // namespace

ConstantVelocityKF::ConstantVelocityKF(double initial_position, double initial_velocity,
                                       std::array<double, 4> initial_covariance, double q,
                                       double r)
    : x0_(initial_position), x1_(initial_velocity), p_(initial_covariance), q_(q), r_(r) {
    if (!finite(initial_position) || !finite(initial_velocity)) {
        throw std::invalid_argument("initial state must be finite");
    }
    if (!(q >= 0.0) || !finite(q)) throw std::invalid_argument("q must be finite and >= 0");
    if (!(r > 0.0) || !finite(r)) throw std::invalid_argument("r must be finite and > 0");
    const auto& p = initial_covariance;
    for (double v : p) {
        if (!finite(v)) throw std::invalid_argument("covariance must be finite");
    }
    if (std::fabs(p[1] - p[2]) > 1e-12 * (1.0 + std::fabs(p[1]))) {
        throw std::invalid_argument("covariance must be symmetric");
    }
    if (p[0] < 0.0 || p[3] < 0.0 || p[0] * p[3] - p[1] * p[2] < -1e-12) {
        throw std::invalid_argument("covariance must be positive semi-definite");
    }
}

void ConstantVelocityKF::predict(double dt) {
    if (!(dt >= 0.0) || !finite(dt)) throw std::invalid_argument("dt must be finite and >= 0");
    if (dt == 0.0) return;
    x0_ += x1_ * dt;
    // P = F P F^T + Q
    const double p00 = p_[0] + dt * (p_[1] + p_[2]) + dt * dt * p_[3];
    const double p01 = p_[1] + dt * p_[3];
    const double p11 = p_[3];
    const double dt2 = dt * dt;
    const double q00 = q_ * dt2 * dt / 3.0;
    const double q01 = q_ * dt2 / 2.0;
    const double q11 = q_ * dt;
    p_ = {p00 + q00, p01 + q01, p01 + q01, p11 + q11};
}

void ConstantVelocityKF::update(double z) {
    if (!finite(z)) throw std::invalid_argument("measurement must be finite");
    const double s = p_[0] + r_;
    const double k0 = p_[0] / s;
    const double k1 = p_[2] / s;
    const double y = z - x0_;
    x0_ += k0 * y;
    x1_ += k1 * y;
    // Joseph form: P = (I - K H) P (I - K H)^T + K r K^T  (keeps P symmetric PSD)
    const double a00 = 1.0 - k0;
    const double a10 = -k1;
    const double m00 = a00 * p_[0];
    const double m01 = a00 * p_[1];
    const double m10 = a10 * p_[0] + p_[2];
    const double m11 = a10 * p_[1] + p_[3];
    const double n00 = m00 * a00 + k0 * r_ * k0;
    const double n01 = m00 * a10 + m01 + k0 * r_ * k1;
    const double n11 = m10 * a10 + m11 + k1 * r_ * k1;
    p_ = {n00, n01, n01, n11};
}

CvState ConstantVelocityKF::state() const { return {x0_, x1_}; }

std::array<double, 4> ConstantVelocityKF::covariance() const { return p_; }

}  // namespace fusion
