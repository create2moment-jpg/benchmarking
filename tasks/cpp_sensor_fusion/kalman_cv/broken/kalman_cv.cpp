#include "kalman_cv.hpp"

#include <cmath>
#include <stdexcept>

namespace fusion {

ConstantVelocityKF::ConstantVelocityKF(double initial_position, double initial_velocity,
                                       std::array<double, 4> initial_covariance, double q,
                                       double r)
    : x0_(initial_position), x1_(initial_velocity), p_(initial_covariance), q_(q), r_(r) {
    if (r <= 0.0) throw std::invalid_argument("r must be > 0");
}

void ConstantVelocityKF::predict(double dt) {
    // Wrong process noise (discrete approximation q*dt on both diagonals) and no dt validation.
    x0_ += x1_ * dt;
    const double p00 = p_[0] + dt * (p_[1] + p_[2]) + dt * dt * p_[3];
    const double p01 = p_[1] + dt * p_[3];
    p_ = {p00 + q_ * dt, p01, p01, p_[3] + q_ * dt};
}

void ConstantVelocityKF::update(double z) {
    const double s = p_[0] + r_;
    const double k0 = p_[0] / s;
    const double k1 = p_[2] / s;
    const double y = z - x0_;
    x0_ += k0 * y;
    x1_ += k1 * y;
    p_ = {(1 - k0) * p_[0], (1 - k0) * p_[1], (1 - k0) * p_[1], p_[3] - k1 * p_[1]};
}

CvState ConstantVelocityKF::state() const { return {x0_, x1_}; }

std::array<double, 4> ConstantVelocityKF::covariance() const { return p_; }

}  // namespace fusion
