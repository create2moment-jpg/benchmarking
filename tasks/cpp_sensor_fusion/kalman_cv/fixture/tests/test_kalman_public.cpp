// Build & run:  g++ -std=c++17 -I. kalman_cv.cpp tests/test_kalman_public.cpp -o kf_public && ./kf_public
#include <cmath>
#include <cstdio>

#include "kalman_cv.hpp"

static int failures = 0;
#define CHECK(name, cond)                                                   \
    do {                                                                    \
        const bool ok_ = (cond);                                            \
        std::printf("CHECK %s %s\n", name, ok_ ? "PASS" : "FAIL");          \
        if (!ok_) ++failures;                                               \
    } while (0)

int main() {
    fusion::ConstantVelocityKF kf(0.0, 1.0, {1.0, 0.0, 0.0, 1.0}, 0.1, 0.5);
    kf.predict(2.0);
    CHECK("predict_moves_position", std::fabs(kf.state().position - 2.0) < 1e-12);
    kf.update(2.0);
    CHECK("update_at_prediction_keeps_position", std::fabs(kf.state().position - 2.0) < 1e-12);
    return failures ? 1 : 0;
}
