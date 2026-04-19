#ifndef COTILA_TENSOR_TEST_H_
#define COTILA_TENSOR_TEST_H_

#include <cotila/cotila.h>

#include <complex>

namespace cotila {
namespace test {

constexpr tensor<double, 1> t1 = {{1.0}};
constexpr tensor<double, 3> t3 = {{1.0, 1.0, 1.0}};
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

static_assert((t1 + 1.0)[0] == 2.0, "tensor + scalar");
static_assert((t3 * 3.0)[1] == 3.0, "tensor * scalar");
static_assert((t31 / 3.0)[1][0] == 1.0, "tensor / scalar");
static_assert((t33 - 3.0)[1][0] == -2.0, "tensor - scalar");

constexpr auto g22 = generate<double, 2, 2>(
    [](std::size_t i, std::size_t j) { return double(i + j); });

static_assert(g22[0][0] == 0.0, "generate[0][0]");
static_assert(g22[0][1] == 1.0, "generate[0][1]");
static_assert(g22[1][0] == 1.0, "generate[1][0]");
static_assert(g22[1][1] == 2.0, "generate[1][1]");

constexpr auto roundtrip = []() {
  tensor<double, 2, 3> r = {};
  set(r, 7.0, 1, 2);
  return at(r, 1, 2);
}();
static_assert(roundtrip == 7.0, "at/set round trip");

static_assert(at(t331, 0, 1, 0) == 2.0, "at on 3-D tensor");
static_assert(at(t331, 2, 2, 0) == 9.0, "at on 3-D tensor");

// Shape helpers.
static_assert(
    std::is_same_v<detail::seq_drop_t<1, 2, 3, 4>, std::index_sequence<2, 4>>,
    "seq_drop middle");
static_assert(
    std::is_same_v<detail::seq_drop_t<0, 2, 3, 4>, std::index_sequence<3, 4>>,
    "seq_drop first");
static_assert(
    std::is_same_v<detail::seq_drop_t<2, 2, 3, 4>, std::index_sequence<2, 3>>,
    "seq_drop last");
static_assert(
    std::is_same_v<detail::seq_cat_t<std::index_sequence<1, 2>,
                                     std::index_sequence<3, 4>>,
                   std::index_sequence<1, 2, 3, 4>>,
    "seq_cat");
static_assert(detail::pack_at<0, 5, 6, 7> == 5, "pack_at 0");
static_assert(detail::pack_at<2, 5, 6, 7> == 7, "pack_at 2");
static_assert(
    std::is_same_v<detail::tensor_from_seq_t<double, std::index_sequence<2, 3>>,
                   tensor<double, 2, 3>>,
    "tensor_from_seq");

}  // namespace test
}  // namespace cotila

#endif  // COTILA_TENSOR_TEST_H_
