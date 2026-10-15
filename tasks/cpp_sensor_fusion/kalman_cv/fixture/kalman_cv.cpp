#include "kalman_cv.hpp"

namespace fusion {

ConstantVelocityKF::ConstantVelocityKF(double initial_position, double initial_velocity,
                                       std::array<double, 4> initial_covariance, double q,
                                       double r)
    : x0_(initial_position), x1_(initial_velocity), p_(initial_covariance), q_(q), r_(r) {}

void ConstantVelocityKF::predict(double dt) {
    // TODO: covariance propagation and process noise are missing.
    x0_ += x1_ * dt;
}

void ConstantVelocityKF::update(double z) {
    const double s = p_[0] + r_;
    const double k0 = p_[0] / s;
    const double k1 = p_[2] / s;
    const double y = z - x0_;
    x0_ += k0 * y;
    x1_ += k1 * y;
    // Simple form (I - K H) P: numerically fragile, loses symmetry.
    p_ = {(1 - k0) * p_[0], (1 - k0) * p_[1], p_[2] - k1 * p_[0], p_[3] - k1 * p_[1]};
}

CvState ConstantVelocityKF::state() const { return {x0_, x1_}; }

std::array<double, 4> ConstantVelocityKF::covariance() const { return p_; }

}  // namespace fusion
