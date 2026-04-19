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
}  // namespace cotila

#endif  // COTILA_TENSOR_MATH_H_
