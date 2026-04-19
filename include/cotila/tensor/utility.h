#ifndef COTILA_TENSOR_UTILITY_H_
#define COTILA_TENSOR_UTILITY_H_

#include <cotila/detail/concept.h>
#include <cotila/tensor/tensor.h>

#include <tuple>

namespace cotila {

/** \addtogroup tensor
 *  @{
 */


template <typename F, typename T, typename... Tensors, std::size_t Dim, 
          typename U =
              std::invoke_result_t<F, T, typename Tensors::value_type...>>
// requires all_same<T, Tensors...>
constexpr tensor<U, Dim> elementwise(F f, const tensor<T, Dim> &t,
                                     const Tensors &...tensors) {
  tensor<U, Dim> op_applied = {};
  for (std::size_t i = 0; i < Dim; ++i)
    op_applied[i] = std::apply(f, std::forward_as_tuple(t[i], tensors[i]...));
  return op_applied;
}


/** @brief applies a function elementwise between many tensors
 *  @param f a function of type F that operates on many scalars of type T and
 * returns a scalar of type U
 *  @param v an N-vector of type T
 *  @param tensors additional tensors of type T with the same shape
 *  @return an tensor of type T with elements described by \f$
 * f\left(\textbf{t}_{ij...}, \ldots\right) \f$
 *
 *  Applies a function elementwise between many tensors.
 */

template <typename F, typename T, typename... Tensors, std::size_t Dim,
          std::size_t... Rest,
          typename U =
              std::invoke_result_t<F, T, typename Tensors::value_type...>>
// requires all_same<T, Tensors...>
constexpr tensor<U, Dim, Rest...> elementwise(F f,
                                              const tensor<T, Dim, Rest...> &t,
                                              const Tensors &...tensors) {
  tensor<U, Dim, Rest...> op_applied = {};
  for (std::size_t i = 0; i < Dim; ++i)
    op_applied[i] = elementwise<F, T, Tensors..., Rest...>(f, t[i], tensors[i]...);
  return op_applied;
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
