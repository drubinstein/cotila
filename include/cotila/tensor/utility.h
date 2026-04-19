#ifndef COTILA_TENSOR_UTILITY_H_
#define COTILA_TENSOR_UTILITY_H_

#include <cotila/detail/concept.h>
#include <cotila/tensor/tensor.h>

#include <tuple>

namespace cotila {

/** \addtogroup tensor
 *  @{
 */


/** @brief applies a function elementwise across tensors of the same shape
 *  @param f a function invocable on scalars of matching types
 *  @param t the first tensor (its shape determines the result shape)
 *  @param tensors zero or more co-tensors of identical shape
 *  @return a tensor with the same shape whose elements are f applied
 *          pointwise at the leaves.
 */
template <typename F, typename T, std::size_t Dim, std::size_t... Rest,
          typename... Tensors,
          typename U = std::invoke_result_t<F, T,
                                            typename Tensors::value_type...>>
constexpr tensor<U, Dim, Rest...>
elementwise(F f, const tensor<T, Dim, Rest...> &t,
            const Tensors &...tensors) {
  tensor<U, Dim, Rest...> out = {};
  for (std::size_t i = 0; i < Dim; ++i) {
    if constexpr (sizeof...(Rest) == 0)
      out[i] = std::apply(f, std::forward_as_tuple(t[i], tensors[i]...));
    else
      out[i] = elementwise(f, t[i], tensors[i]...);
  }
  return out;
}

template <typename U, typename F, typename T, std::size_t Dim, typename... Args>
// requires all_same<std::size_t, Args...>
constexpr decltype(auto) generate(F &&f, Args... args) {
  tensor<U, Dim> generated = {};
  for (std::size_t i = 0; i < Dim; ++i) {
    generated[i] = std::apply(f, std::forward_as_tuple(args...));
  }
}

template <typename U, typename F, typename T, std::size_t Dim,
          std::size_t... Rest, typename... Args>
//requires all_same<std::size_t, Args...>
constexpr decltype(auto) generate(F &&f, Args... args) {
  tensor<U, Dim, Rest...> generated = {};
  for (std::size_t i = 0; i < Dim; ++i) {
    generated[i] = generate<U, F, T, Rest...>(f, Dim, args...);
  }
  return generated;
}

/** @brief generates a tensor as a function of its indices
 *  @param f a function that operates on two integer indices
 *  @return a tensor with type matching the return type of f such that \f$
 * \textbf{t}_{ij...} = f(i, j, ...) \f$
 *
 *  Generates a tensor as a function of its indices.
 */
template <typename U, typename F, typename T, std::size_t Dim,
          std::size_t... Rest>
constexpr decltype(auto) generate(F &&f) {
  tensor<U, Dim, Rest...> generated = {};
  for (std::size_t i = 0; i < Dim; ++i) {
    generated[i] = generate<U, F, T, Rest...>(f, Dim);
  }
  return generated;
}

}  // namespace cotila

#endif  // COTILA_TENSOR_UTILITY_H_
