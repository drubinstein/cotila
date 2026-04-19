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

// tensordot<1, 0> is matmul. A is 2x3, B is 3x2. Expected result is 2x2.
constexpr tensor<double, 2, 3> A = {{
    {1.0, 2.0, 3.0},
    {4.0, 5.0, 6.0}
}};
constexpr tensor<double, 3, 2> B = {{
    {7.0, 8.0},
    {9.0, 10.0},
    {11.0, 12.0}
}};
constexpr auto AB = tensordot<1, 0>(A, B);
static_assert(AB[0][0] == 1*7 + 2*9 + 3*11, "matmul [0][0]");
static_assert(AB[0][1] == 1*8 + 2*10 + 3*12, "matmul [0][1]");
static_assert(AB[1][0] == 4*7 + 5*9 + 6*11, "matmul [1][0]");
static_assert(AB[1][1] == 4*8 + 5*10 + 6*12, "matmul [1][1]");

constexpr tensor<double, 3> u = {{1.0, 2.0, 3.0}};
constexpr tensor<double, 3> v = {{4.0, 5.0, 6.0}};
static_assert(tensordot<0, 0>(u, v) == 1*4 + 2*5 + 3*6, "vector dot");

// Contract axis 2 of a (size 4) with axis 0 of b (size 4).
// Result shape: (2, 3) ++ (5,) = (2, 3, 5).
constexpr tensor<double, 2, 3, 4> C = generate<double, 2, 3, 4>(
    [](std::size_t i, std::size_t j, std::size_t k) {
      return double(i * 100 + j * 10 + k);
    });
constexpr tensor<double, 4, 5> D = generate<double, 4, 5>(
    [](std::size_t k, std::size_t l) { return double(k * 10 + l); });
constexpr auto CD = tensordot<2, 0>(C, D);
static_assert(
    std::is_same_v<decltype(CD), const tensor<double, 2, 3, 5>>,
    "tensordot<2,0> result shape");

// Spot-check: CD[0][0][0] = sum_k C[0][0][k] * D[k][0]
//                       = 0*0 + 1*10 + 2*20 + 3*30 = 140
static_assert(CD[0][0][0] == 140.0, "tensordot<2,0>[0][0][0]");

// Spot-check: CD[1][2][4] = sum_k C[1][2][k] * D[k][4]
//                       = (100+20+0)*4 + (100+20+1)*14 + (100+20+2)*24 + (100+20+3)*34
//                       = 120*4 + 121*14 + 122*24 + 123*34
//                       = 480 + 1694 + 2928 + 4182 = 9284
static_assert(CD[1][2][4] == 9284.0, "tensordot<2,0>[1][2][4]");

// Symmetric scalars (operator with scalar on the left).
static_assert((1.0 + t1)[0] == 2.0, "scalar + tensor");
static_assert((3.0 * t3)[1] == 3.0, "scalar * tensor");
static_assert((6.0 / t31)[1][0] == 2.0, "scalar / tensor");
static_assert((3.0 - t33)[1][0] == 2.0, "scalar - tensor");

// Elementwise tensor-tensor.
constexpr tensor<double, 2, 2> P = {{
    {1.0, 2.0},
    {3.0, 4.0}
}};
constexpr tensor<double, 2, 2> Q = {{
    {5.0, 6.0},
    {7.0, 8.0}
}};
static_assert((P + Q)[0][0] == 6.0, "tensor + tensor rank-2");
static_assert((P + Q)[1][1] == 12.0, "tensor + tensor rank-2");
static_assert((P - Q)[1][0] == -4.0, "tensor - tensor rank-2");
static_assert((P * Q)[0][1] == 12.0, "tensor * tensor rank-2 (hadamard)");
static_assert((Q / P)[0][0] == 5.0, "tensor / tensor rank-2");

constexpr tensor<double, 2, 2, 2> R1 = {{
    {{ {1.0, 2.0}, {3.0, 4.0} }},
    {{ {5.0, 6.0}, {7.0, 8.0} }}
}};
constexpr tensor<double, 2, 2, 2> R2 = {{
    {{ {1.0, 1.0}, {1.0, 1.0} }},
    {{ {1.0, 1.0}, {1.0, 1.0} }}
}};
static_assert((R1 + R2)[0][1][1] == 5.0, "tensor + tensor rank-3");
static_assert((R1 - R2)[1][0][0] == 4.0, "tensor - tensor rank-3");

// Equality.
static_assert(P == P, "tensor == self");
static_assert(!(P == Q), "tensor != other");
static_assert(P != Q, "operator!=");

}  // namespace test
}  // namespace cotila

#endif  // COTILA_TENSOR_TEST_H_
