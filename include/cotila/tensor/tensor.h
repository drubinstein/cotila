/** @file
 *  @brief Contains the definition of the `cotila::tensor` class.
 */
#pragma once
#ifndef COTILA_TENSOR_TENSOR_H_
#define COTILA_TENSOR_TENSOR_H_

#include <cotila/detail/assert.h>
#include <cotila/detail/tmp.h>
#include <cotila/vector/vector.h>

#include <cstdint>
#include <type_traits>

namespace cotila {

template <typename T, std::size_t Dimension, std::size_t... Shape>
struct tensor {
  // TODO: Add an assertion for non as 0
  COTILA_DETAIL_ASSERT_ARITHMETIC(T)

  using value_type = T;
  using size_type = std::size_t;

  constexpr tensor<T, Shape...> operator[](std::size_t i) noexcept {
    return array[i];
  }
  constexpr const tensor<T, Shape...> operator[](size_type i) const noexcept {
    return array[i];
  }

  constexpr tensor<T, Shape...> *begin() noexcept { return array; }
  constexpr tensor<T, Shape...> *end() noexcept { return array + Dimension; }
  constexpr const tensor<T, Shape...> *cbegin() const noexcept { return array; }
  constexpr const tensor<T, Shape...> *cend() const noexcept { return array + Dimension; }

  tensor<T, Shape...> array[Dimension];  ///< @private
};

template <typename T, std::size_t Dimension>
struct tensor<T, Dimension> {
  // TODO: Add an assertion for non as 0
  COTILA_DETAIL_ASSERT_ARITHMETIC(T)

  using value_type = T;
  using size_type = std::size_t;
  constexpr T operator[](std::size_t i) noexcept { return array[i]; }
  constexpr const T operator[](size_type i) const noexcept { return array[i]; }
  
  constexpr T *begin() noexcept { return array; }
  constexpr T *end() noexcept { return array + Dimension; }
  constexpr const T *cbegin() const noexcept { return array; }
  constexpr const T *cend() const noexcept { return array + Dimension; }

  T array[Dimension];  ///< @private
};

}  // namespace cotila

#endif  // COTILA_TENSOR_TENSOR_H_