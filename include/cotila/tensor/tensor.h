/** @file
 *  @brief Contains the definition of the `cotila::tensor` class.
 */
#pragma once
#ifndef COTILA_TENSOR_TENSOR_H_
#define COTILA_TENSOR_TENSOR_H_

#include <cotila/detail/tmp.h>
#include <cotila/detail/assert.h>
#include <cstdint>
#include <type_traits>

namespace cotila {

template <typename T, std::size_t... Dimension>
struct tensor_impl;

template <typename T, std::size_t... Dimension>
using tensor_impl_t = typename tensor_impl<T, Dimension...>::type;

template <typename T>
struct tensor_impl<T> {
    using type = T;
};

template <typename T, std::size_t First, std::size_t... Dimension>
struct tensor_impl<T, First, Dimension...> {
    using type = tensor_impl_t<T, Dimension...>[First];
};

/** @brief A container representing a tensor
 *  @tparam T scalar type to contain
 *  @tparam Shape The shape of the tensor
 * 
 *  `cotila::tensor` is a container representing a tensor.
 * 
 *  It is an aggregate type more similar, but more abstract than `cotila::matrix`
 *  and can be initialized with aggregate initialization 
 *  or with the `cotila::make_tensor` function.
 */
 template <typename T, std::size_t... Shape> struct tensor {
   // TODO: Add an assertion for non as 0
   COTILA_DETAIL_ASSERT_ARITHMETIC(T)

   using value_type = T;
   using size_type = std::size_t;
   static constexpr size_type shape[sizeof...(Shape)] = {Shape...};

   
  /** @name Access specific element from leading dimension */
  ///@{
  /** @brief Access specified dimension
   *  @param i index of the slice to extract. Must be same rank as the shape.
   *  @return The selected element from the leading dimension
   *
   *  Given a Tensor with shape (x,y,z,...) return T[i][y][z]...
   */
    constexpr auto operator[](std::size_t i) -> auto { return arrays[i]; }

   tensor_impl<T, Shape...> arrays;
 };

template <typename T, std::size_t... Shape>
tensor(const tensor_impl<T, Shape...>)->tensor<T, Shape...>;

} // namespace cotila

#endif // COTILA_TENSOR_TENSOR_H_