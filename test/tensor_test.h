#ifndef COTILA_TENSOR_TEST_H_
#define COTILA_TENSOR_TEST_H_

#include <cotila/cotila.h>

#include <complex>

namespace cotila {
namespace test {

// Uniform initialization
constexpr tensor<double, 1> t1 = {{1.0}};
constexpr tensor<double, 3> t3 = {{1.0, 1.0, 1.0}};
constexpr tensor<double, 3, 1> t31= {{1.0, 1.0, 1.0}};
constexpr tensor<double, 3, 3> t33 = {{
    {1.0, 1.0, 1.0},
    {1.0, 1.0, 1.0},
    {1.0, 1.0, 1.0}
}};
constexpr tensor<double, 1, 3, 3> t133 = {{
    {{
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0}
    }}
}};

constexpr tensor<double, 3, 3, 1> t331 = {{
    {1., 1., 1.}, 
    {1., 1., 1.}, 
    {1., 1., 1.}
}};

}  // namespace test
}  // namespace cotila

#endif  // COTILA_TENSOR_TEST_H_
