// 1-D constant-velocity Kalman filter used to smooth a tracked object's longitudinal position.
#pragma once

#include <array>

namespace fusion {

struct CvState {
    double position;
    double velocity;
};

// State x = [position, velocity]^T, covariance P stored row-major {P00, P01, P10, P11}.
// Process model: continuous white-noise acceleration with spectral density q:
//   F = [[1, dt], [0, 1]],  Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]
// Measurement model: position only, z = H x + v,  H = [1, 0],  var(v) = r.
class ConstantVelocityKF {
public:
    ConstantVelocityKF(double initial_position, double initial_velocity,
                       std::array<double, 4> initial_covariance, double q, double r);

    void predict(double dt);
    void update(double z);

    CvState state() const;
    std::array<double, 4> covariance() const;

private:
    double x0_;
    double x1_;
    std::array<double, 4> p_;
    double q_;
    double r_;
};

}  // namespace fusion
