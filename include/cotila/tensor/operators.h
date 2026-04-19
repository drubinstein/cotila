#ifndef COTILA_tensor_OPERATORS_H_
#define COTILA_tensor_OPERATORS_H_

#include <cotila/tensor/tensor.h>

namespace cotila {

// TODO: REPAIR COMMENTS

/** \addtogroup tensor
 *  @{
 */

/** @brief checks equality of two tensors
 *  @param a a tensor of type T
 *  @param b a tensor of type T
 *  @return true if and only every element of a and b match.$
 *
 *  Checks the equality of two tensors.
 */
template <typename T, std::size_t Dim, std::size_t... Rest>
constexpr bool operator==(const tensor<T, Dim, Rest...> &a,
                          const tensor<T, Dim, Rest...> &b) {
  for (std::size_t i = 0; i < Dim; ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/** @brief checks inequality of two tensors
 *  @param a a tensor of type T
 *  @param b a tensor of type T
 *  @return true if and only every element of a and b \it{do not} match.$
 *
 *  Checks the inequality of two tensors.
 */
template <typename T, std::size_t... N>
constexpr bool operator!=(const tensor<T, N...> &a, const tensor<T, N...> &b) {
  return !(a == b);
}

/** @brief computes the sum of a tensor and a scalar
 *  @param m a tensor of type T
 *  @param a a scalar of type T
 *  @return A new tensor such that each element is the sum of the corresponding
 *  element of m + a.
 *
 *  Computes the sum of a tensor and a scalar.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator+(const tensor<T, N...> &m, T a) {
  return elementwise([a](T x) { return x + a; }, m);
}

/** @brief computes the sum of a tensor and a scalar
 *  @param a a scalar of type T
 *  @param m a tensor of type T
 *  @return A new tensor such that each element is the sum of the corresponding
 *  element of m + a.
 *
 *  Computes the sum of a tensor and a scalar.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator+(T a, const tensor<T, N...> &m) {
  return m + a;
}

/** @brief computes elementwise the sum of two tensors
 *  @param a a tensor of type T
 *  @param b a tensor of type T
 *  @return A new tensor such that each element is the sum of corresponding
 *  elements of a and b.
 *
 *  Computes the elementwise sum.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator+(const tensor<T, N...> &a,
                                    const tensor<T, N...> &b) {
  return elementwise(std::plus<T>(), a, b);
}

/** @brief computes the product of a tensor and a scalar
 *  @param m a tensor of type T
 *  @param a a scalar of type T
 *  @return A new tensor such that each element is the product of the corresponding
 *  element of m * a.
 *
 *  Computes the product of a tensor and a scalar.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator*(const tensor<T, N...> &m, T a) {
  return elementwise([a](T x) { return x * a; }, m);
}

/** @brief computes the product of a tensor and a scalar
 *  @param a a scalar of type T
 *  @param m a tensor of type T
 *  @return A new tensor such that each element is the product of the corresponding
 *  element of m * a.
 *
 *  Computes the product of a tensor and a scalar.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator*(T a, const tensor<T, N...> &m) {
  return m * a;
}

/** @brief computes the Hadamard product
 *  @param A a tensor of type T
 *  @param b a tensor of type T
 *  @return A new tensor such that each element is the product of corresponding
 *  elements of a and b.
 *
 *  Computes the Hadamard, or elementwise, product of two tensors.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator*(const tensor<T, N...> &a,
                                    const tensor<T, N...> &b) {
  return elementwise(std::multiplies<T>(), a, b);
}

/** @brief tensor / scalar (elementwise) */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator/(const tensor<T, N...> &m, T a) {
  return elementwise([a](T x) { return x / a; }, m);
}

/** @brief computes the quotient of a tensor and a scalar
 *  @param m a tensor of type T
 *  @param a a scalar of type T
 *  @return A new tensor such that each element is the quotient of the corresponding
 *  element of m * a.
 *
 *  Computes the quotient of a tensor and a scalar.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator/(T a, const tensor<T, N...> &m) {
  return elementwise([a](T x) { return a / x; }, m);
}

/** @brief computes the quotient of a tensor and a scalar
 *  @param a a scalar of type T
 *  @param m a tensor of type T
 *  @return A new tensor such that each element is the quotient of the corresponding
 *  element of m * a.
 *
 *  Computes the quotient of a tensor and a scalar.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator/(const tensor<T, N...> &a,
                                    const tensor<T, N...> &b) {
  return elementwise(std::divides<T>(), a, b);
}

/** @brief tensor - scalar (elementwise) */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator-(const tensor<T, N...> &m, T a) {
  return elementwise([a](T x) { return x - a; }, m);
}

/** @brief scalar - tensor (elementwise) */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator-(T a, const tensor<T, N...> &m) {
  return elementwise([a](T x) { return a - x; }, m);
}

/** @brief tensor - tensor (elementwise) */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator-(const tensor<T, N...> &a,
                                    const tensor<T, N...> &b) {
  return elementwise(std::minus<T>(), a, b);
}

/** }@*/

}  // namespace cotila

#endif  // COTILA_VECTOR_OPERATORS_H_
