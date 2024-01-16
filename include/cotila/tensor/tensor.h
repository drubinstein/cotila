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
#include <tuple>
#include <type_traits>
#include <utility>

namespace cotila {

template <std::size_t...>
struct shape {};

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
  constexpr const tensor<T, Shape...> *cend() const noexcept {
    return array + Dimension;
  }

  template <std::size_t Begin, std::size_t... Begins, std::size_t Size,
            std::size_t... Sizes>
  constexpr auto slice(const shape<Begin, Begins...>,
                       const shape<Size, Sizes...>) const noexcept {
    tensor<T, Size, Sizes...> sliced;
    std::transform(
        array + Begin, array + Begin + Size, sliced.array,
        [](auto x) { return x.slice(shape<Begins...>(), shape<Sizes...>()); });
    return sliced;
  }

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

  template <std::size_t Begin, std::size_t Size>
  constexpr auto slice(const shape<Begin>, const shape<Size>) const noexcept {
    tensor<T, Size> sliced;
    std::copy_n(array + Begin, Size, sliced.array);
    return sliced;
  }

  T array[Dimension];  ///< @private
};

/** \addtogroup tensor
 *  @{
 */

/** @name cotila::tensor deduction guides */
///@{

// TODO: Can this be done with a variadic macro?
/** @brief deduction guide for aggregate initialization
 *  @relatesalso cotila::tensor
 *
 *  This deduction guide allows cotila::tensor to be constructed like this:
 *  \code{.cpp}
 *  cotila::tensor m{{{1., 2.}, {3., 4.}}}; // deduces the type of m to be
 * cotila::tensor<double, 2, 2> \endcode
 */
template <typename T, std::size_t M>
tensor(const T (&)[M]) -> tensor<T, M>;

template <typename T, std::size_t M, std::size_t N>
tensor(const T (&)[M][N]) -> tensor<T, M, N>;

template <typename T, std::size_t M, std::size_t N, std::size_t O>
tensor(const T (&)[M][N][O]) -> tensor<T, M, N, O>;

///@}

/** @}*/

}  // namespace cotila

#endif  // COTILA_TENSOR_TENSOR_H_