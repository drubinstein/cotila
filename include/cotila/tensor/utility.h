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

/** @brief builds a tensor of the requested shape from a function of indices
 *  @param f invocable as f(i0, i1, ...) returning T, one arg per dimension
 *  @return a tensor where element [i0][i1]...[ik] equals f(i0, i1, ..., ik)
 */
template <typename T, std::size_t Dim, std::size_t... Rest,
          typename F, typename... Idx>
constexpr tensor<T, Dim, Rest...> generate(F f, Idx... prior) {
  tensor<T, Dim, Rest...> out = {};
  for (std::size_t i = 0; i < Dim; ++i) {
    if constexpr (sizeof...(Rest) == 0)
      out[i] = f(prior..., i);
    else
      out[i] = generate<T, Rest...>(f, prior..., i);
  }
  return out;
}

}  // namespace cotila

#endif  // COTILA_TENSOR_UTILITY_H_
