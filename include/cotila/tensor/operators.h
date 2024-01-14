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
template <typename T, std::size_t N, std::size_t... M>
constexpr bool operator==(const tensor<T, N, M...> &a,
                          const tensor<T, N, M...> &b) {
  for (std::size_t i = 0; i < N; ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

template <typename T, std::size_t N>
constexpr bool operator==(const tensor<T, N> &a, const tensor<T, N> &b) {
  for (std::size_t i = 0; i < N; ++i) {
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
 *  @param m an tensor of type T
 *  @param a a scalar of type T
 *  @return \f$ \textbf{m} + a \f$ such that \f$ \left(\textbf{m} +
 * a\right)_{ij} = \textbf{m}_{ij} + a \f$
 *
 *  Computes the sum of a tensor and a scalar.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator+(const tensor<T, N...> &m, T a) {
  return elementwise([a](T x) { return x + a; }, m);
}

/** @brief computes the sum of a tensor and a scalar
 *  @param a a scalar of type T
 *  @param m an \f$ N \times M \f$ tensor of type T
 *  @return \f$ a + \textbf{m} \f$ such that \f$ \left(a +
 * \textbf{m}\right)_{ij} = a + \textbf{m}_{ij} \f$
 *
 *  Computes the sum of a tensor and a scalar.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator+(T a, const tensor<T, N...> &m) {
  return m + a;
}

/** @brief computes the tensor sum
 *  @param a an \f$ N \times M \f$ tensor of type T
 *  @param b an \f$ N \times M \f$ tensor of type T
 *  @return \f$ \textbf{a} + \textbf{b} \f$ such that \f$ \left(\textbf{a} +
 * \textbf{b}\right)_{ij} = \textbf{a}_{ij} + \textbf{b}_{ij} \f$
 *
 *  Computes the vector sum.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator+(const tensor<T, N...> &a,
                                    const tensor<T, N...> &b) {
  return elementwise(std::plus<T>(), a, b);
}

/** @brief computes the product of a tensor and a scalar
 *  @param m an \f$ N \times M \f$ tensor of type T
 *  @param a a scalar of type T
 *  @return \f$ \textbf{m}a \f$ such that \f$ \left(\textbf{m} a\right)_{ij} =
 * \textbf{m}_{ij} a \f$
 *
 *  Computes the sum of a tensor and a scalar.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator*(const tensor<T, N...> &m, T a) {
  return elementwise([a](T x) { return x * a; }, m);
}

/** @brief computes the product of a tensor and a scalar
 *  @param a a scalar of type T
 *  @param m an \f$ N \times M \f$ tensor of type T
 *  @return \f$ a\textbf{m} \f$ such that \f$ \left(a\textbf{m}\right)_{ij} =
 * a\textbf{m}_{ij} \f$
 *
 *  Computes the sum of a tensor and a scalar.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator*(T a, const tensor<T, N...> &m) {
  return m * a;
}

/** @brief computes the Hadamard product
 *  @param a an \f$ N \times M \f$ tensor of type T
 *  @param b an \f$ N \times M \f$ tensor of type T
 *  @return \f$ \textbf{a} \circ \textbf{b} \f$ such that \f$ \left(\textbf{a}
 * \circ \textbf{b}\right)_{ij} = \textbf{a}_{ij} \textbf{b}_{ij} \f$
 *
 *  Computes the Hadamard, or elementwise, product of two vectors.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator*(const tensor<T, N...> &a,
                                    const tensor<T, N...> &b) {
  return elementwise(std::multiplies<T>(), a, b);
}

/** @brief computes the quotient between a tensor and a scalar
 *  @param m an \f$ N \times M \f$ tensor of type T
 *  @param a a scalar of type T
 *  @return \f$ \textbf{m}/a \f$ such that \f$ \left(\textbf{m}/a\right)_{ij} =
 * \frac{\textbf{m}_{ij}}{a} \f$
 *
 *  Computes division between a tensor and a scalar.
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator/(T a, const tensor<T, N...> &m) {
  return elementwise([a](T x) { return a / x; }, m);
}

/** @brief computes the elementwise tensor quotient
 *  @param a an \f$ N \times M \f$ tensor of type T
 *  @param b an \f$ N \times M \f$ tensor of type T
 *  @return \f$ \textbf{a} \circ \textbf{b}' \f$ such that \f$
 * {\textbf{b}_{ij}}' = \left(\textbf{b}_{ij}\right)^{-1}\f$ and \f$
 * \left(\textbf{a} \circ \textbf{b}'\right)_{ij} = \textbf{a}_{ij}
 * {\textbf{b}'}_{ij} \f$
 *
 *  Computes elementwise division between two tensors
 */
template <typename T, std::size_t... N>
constexpr tensor<T, N...> operator/(const tensor<T, N...> &a,
                                    const tensor<T, N...> &b) {
  return elementwise(std::divides<T>(), a, b);
}

/** }@*/

}  // namespace cotila

#endif  // COTILA_VECTOR_OPERATORS_H_
