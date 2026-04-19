/** @file
 *  @brief Mathematical operations on tensors.
 */
#ifndef COTILA_TENSOR_MATH_H_
#define COTILA_TENSOR_MATH_H_

#include <cotila/tensor/tensor.h>
#include <cotila/tensor/utility.h>

#include <array>
#include <utility>

namespace cotila {
namespace detail {

// Concatenate two index_sequences.
template <typename A, typename B> struct seq_cat;
template <std::size_t... A, std::size_t... B>
struct seq_cat<std::index_sequence<A...>, std::index_sequence<B...>> {
  using type = std::index_sequence<A..., B...>;
};
template <typename A, typename B>
using seq_cat_t = typename seq_cat<A, B>::type;

// Drop element at position I from a pack of std::size_t values.
template <std::size_t I, std::size_t... S> struct seq_drop {
 private:
  static constexpr std::size_t N = sizeof...(S);
  static_assert(I < N, "seq_drop: index out of range");
  static constexpr std::array<std::size_t, N> arr = {S...};
  template <std::size_t... Head, std::size_t... Tail>
  static constexpr auto build(std::index_sequence<Head...>,
                              std::index_sequence<Tail...>)
      -> std::index_sequence<arr[Head]..., arr[I + 1 + Tail]...>;

 public:
  using type = decltype(build(std::make_index_sequence<I>{},
                              std::make_index_sequence<N - I - 1>{}));
};
template <std::size_t I, std::size_t... S>
using seq_drop_t = typename seq_drop<I, S...>::type;

// Materialize a tensor type from an index_sequence.
template <typename T, typename Seq> struct tensor_from_seq;
template <typename T, std::size_t... S>
struct tensor_from_seq<T, std::index_sequence<S...>> {
  using type = tensor<T, S...>;
};
template <typename T, typename Seq>
using tensor_from_seq_t = typename tensor_from_seq<T, Seq>::type;

// Value-level access to a pack element (constexpr).
template <std::size_t I, std::size_t... S>
constexpr std::size_t pack_at = std::array<std::size_t, sizeof...(S)>{S...}[I];

}  // namespace detail

namespace detail {

// Increment a multi-index odometer (LSB = last axis). Returns false when wrapped.
template <std::size_t N>
constexpr bool inc_odometer(std::array<std::size_t, N> &idx,
                            const std::array<std::size_t, N> &shape) {
  for (std::size_t k = N; k-- > 0;) {
    if (++idx[k] < shape[k]) return true;
    idx[k] = 0;
  }
  return false;
}

// Insert `v` at position P in an array of size N (producing size N+1).
template <std::size_t P, std::size_t N>
constexpr std::array<std::size_t, N + 1>
insert_at(const std::array<std::size_t, N> &a, std::size_t v) {
  std::array<std::size_t, N + 1> out = {};
  for (std::size_t i = 0; i < P; ++i) out[i] = a[i];
  out[P] = v;
  for (std::size_t i = P; i < N; ++i) out[i + 1] = a[i];
  return out;
}

// Apply std::array<std::size_t, N> as N separate arguments to `at` / `set`.
template <typename T, std::size_t Dim, std::size_t... Rest, std::size_t N,
          std::size_t... Ks>
constexpr decltype(auto)
at_array_impl(const tensor<T, Dim, Rest...> &t,
              const std::array<std::size_t, N> &idx,
              std::index_sequence<Ks...>) {
  return at(t, idx[Ks]...);
}
template <typename T, std::size_t Dim, std::size_t... Rest, std::size_t N>
constexpr decltype(auto) at_array(const tensor<T, Dim, Rest...> &t,
                                  const std::array<std::size_t, N> &idx) {
  static_assert(N == 1 + sizeof...(Rest), "index rank mismatch");
  return at_array_impl(t, idx, std::make_index_sequence<N>{});
}

template <typename T, std::size_t Dim, std::size_t... Rest, std::size_t N,
          std::size_t... Ks>
constexpr void set_array_impl(tensor<T, Dim, Rest...> &t, T v,
                              const std::array<std::size_t, N> &idx,
                              std::index_sequence<Ks...>) {
  set(t, v, idx[Ks]...);
}
template <typename T, std::size_t Dim, std::size_t... Rest, std::size_t N>
constexpr void set_array(tensor<T, Dim, Rest...> &t, T v,
                         const std::array<std::size_t, N> &idx) {
  static_assert(N == 1 + sizeof...(Rest), "index rank mismatch");
  set_array_impl(t, v, idx, std::make_index_sequence<N>{});
}

}  // namespace detail

/** @brief contract two tensors over one axis each
 *  @tparam AxisA index of the contracted axis in `a`
 *  @tparam AxisB index of the contracted axis in `b`
 *  @return a tensor whose shape is a's shape with AxisA dropped, followed
 *          by b's shape with AxisB dropped. Elements are
 *          sum_c a[...,c@AxisA,...] * b[...,c@AxisB,...].
 *
 *  The contracted-axis sizes of a and b must be equal — checked via
 *  static_assert.
 */
template <std::size_t AxisA, std::size_t AxisB,
          typename T, std::size_t... As, std::size_t... Bs>
constexpr auto tensordot(const tensor<T, As...> &a,
                         const tensor<T, Bs...> &b) {
  static_assert(AxisA < sizeof...(As), "AxisA out of range");
  static_assert(AxisB < sizeof...(Bs), "AxisB out of range");
  static_assert(detail::pack_at<AxisA, As...> == detail::pack_at<AxisB, Bs...>,
                "contracted axes must have equal size");

  using ResSeq = detail::seq_cat_t<detail::seq_drop_t<AxisA, As...>,
                                   detail::seq_drop_t<AxisB, Bs...>>;
  using Result = detail::tensor_from_seq_t<T, ResSeq>;

  constexpr std::size_t ContractSize = detail::pack_at<AxisA, As...>;
  constexpr std::size_t RankA = sizeof...(As);
  constexpr std::size_t RankB = sizeof...(Bs);
  constexpr std::array<std::size_t, RankA> shape_a = {As...};
  constexpr std::array<std::size_t, RankB> shape_b = {Bs...};

  std::array<std::size_t, RankA - 1> shape_a_rem = {};
  for (std::size_t i = 0, j = 0; i < RankA; ++i)
    if (i != AxisA) shape_a_rem[j++] = shape_a[i];
  std::array<std::size_t, RankB - 1> shape_b_rem = {};
  for (std::size_t i = 0, j = 0; i < RankB; ++i)
    if (i != AxisB) shape_b_rem[j++] = shape_b[i];

  Result out = {};
  std::array<std::size_t, RankA - 1> ia = {};
  do {
    std::array<std::size_t, RankB - 1> ib = {};
    do {
      T acc = T{};
      for (std::size_t c = 0; c < ContractSize; ++c) {
        auto a_idx = detail::insert_at<AxisA>(ia, c);
        auto b_idx = detail::insert_at<AxisB>(ib, c);
        acc += detail::at_array(a, a_idx) * detail::at_array(b, b_idx);
      }
      std::array<std::size_t, (RankA - 1) + (RankB - 1)> out_idx = {};
      for (std::size_t i = 0; i < RankA - 1; ++i) out_idx[i] = ia[i];
      for (std::size_t i = 0; i < RankB - 1; ++i) out_idx[(RankA - 1) + i] = ib[i];
      detail::set_array(out, acc, out_idx);
    } while (detail::inc_odometer(ib, shape_b_rem));
  } while (detail::inc_odometer(ia, shape_a_rem));
  return out;
}

}  // namespace cotila

#endif  // COTILA_TENSOR_MATH_H_
