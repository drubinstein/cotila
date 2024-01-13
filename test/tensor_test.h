#ifndef COTILA_TENSOR_TEST_H_
#define COTILA_TENSOR_TEST_H_

#include <complex>
#include <cotila/cotila.h>

namespace cotila {
namespace test {

// Uniform initialization
constexpr tensor<double, 1, 3, 3> t1 = {
    {1., 2., 3.,
    4., 5., 6.,
    7., 8., 9.}};


} // namespace test
} // namespace cotila

#endif // COTILA_TENSOR_TEST_H_
