// g++ -std=c++17 -I. association.cpp tests/test_association_public.cpp -o assoc_public && ./assoc_public
#include <cstdio>

#include "association.hpp"

int main() {
    const std::vector<double> cost = {1.0, 9.0, 9.0, 2.0};
    const auto a = fusion::greedy_associate(cost, 2, 2, 5.0);
    const bool ok = a.size() == 2 && a[0].detection == 0 && a[1].detection == 1;
    std::printf("CHECK diagonal %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
