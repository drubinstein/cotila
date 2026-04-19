#ifndef COTILA_TENSOR_TEST_H_
#define COTILA_TENSOR_TEST_H_

#include <cotila/cotila.h>

#include <complex>

namespace cotila {
namespace test {

constexpr tensor<double, 1> t1 = {{1.0}};
constexpr tensor<double, 3> t3 = {{1.0, 1.0, 1.0}};

static_assert((t1 + 1.0)[0] == 2.0, "tensor + scalar");
static_assert((t3 * 3.0)[1] == 3.0, "tensor * scalar");

// Remaining tests re-enabled progressively.
#if 0
constexpr tensor<double, 3, 1> t31 = {{1.0, 3.0, 1.0}};
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
    {1., 2., 3.},
    {4., 5., 6.},
    {7., 8., 9.}
}};

constexpr auto t331s = t331.slice(shape<1, 1, 0>(), shape<1, 1, 1>());

static_assert(t331s[0][0][0] == 5.0, "tensor slice");

static_assert((t31 / 3.0)[1][0] == 1.0, "tensor / scalar");
static_assert((t33 - 3.0)[1][0] == -2.0, "tensor - scalar");
#endif

}  // namespace test
}  // namespace cotila

#endif  // COTILA_TENSOR_TEST_H_
