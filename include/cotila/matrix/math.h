/** @file
 *  @brief Mathematical operations on matrices.
 */
#ifndef COTILA_MATRIX_MATH_H_
#define COTILA_MATRIX_MATH_H_

#include<algorithm>

#include <cotila/scalar/math.h>
#include <cotila/vector/vector.h>
#include <cotila/vector/math.h>
#include <cotila/matrix/matrix.h>
#include <cotila/matrix/utility.h>

namespace cotila {

/** @brief computes the elementwise complex conjugate
 *  @param m an \f$ M \times N \f$ matrix of type T
 *  @return an \f$ M \times N \f$ matrix \f$\overline{\textbf{m}}\f$ of type T such that
 *  \f$ \left(\overline{\textbf{m}}\right)_{ij} = \overline{\textbf{m}_{ij}},\ \forall i,j \f$
 *
 *  Computes the elementwise complex conjugate of a matrix
 */
template <typename T, std::size_t M, std::size_t N>
constexpr matrix<T, M, N> conj(const matrix<T, M, N> &m) {
  return elementwise(cotila::conj<T>, m);
}

/** @brief computes the transpose
 *  @param m an \f$ M \times N \f$ matrix of type T
 *  @return an \f$ N \times M \f$ matrix \f$ \textbf{m}^{\mathrm{T}} \f$ of type T such that
 *  \f$ \left(\textbf{m}^{\mathrm{T}}\right)_{ij} = \textbf{m}_{ji},\ \forall i,j \f$
 *
 *  Computes the matrix transpose.
 */
template <typename T, std::size_t M, std::size_t N>
constexpr matrix<T, N, M> transpose(const matrix<T, M, N> &m) {
  return generate<N, M>([&m](auto i, auto j){return m[j][i];});
}

/** @brief computes the Hermitian transpose
 *  @param m an \f$ M \times N \f$ matrix of type T
 *  @return an \f$ N \times M \f$ matrix \f$ \textbf{m}^{*} \f$ of type T such that
 *  \f$ \left(\textbf{m}^{*}\right)_{ij} = \overline{\textbf{m}_{ji}},\ \forall i,j \f$
 *
 *  Computes the Hermitian (conjugate) transpose.
 */
template <typename T, std::size_t M, std::size_t N>
constexpr matrix<T, N, M> hermitian(const matrix<T, M, N> &m) {
  return transpose(conj(m));
}

/** @brief computes the matrix product
 *  @param a an \f$M \times N\f$ matrix
 *  @param b an \f$N \times P\f$ matrix
 *  @return an \f$ M \times P \f$ matrix \f$ \textbf{a}\textbf{b} \f$ of type T such that
 *  \f$ \left(\textbf{ab}\right)_{ij} = \sum\limits_{k=1}^{N}\textbf{a}_{ik}\textbf{b}_{kj} \f$
 *
 *  Computes the product of two matrices.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
constexpr matrix<T, M, P> matmul(const matrix<T, M, N> &a,
                                 const matrix<T, N, P> &b) {
  return generate<M, P>([&a, &b](auto i, auto j){return cotila::sum(a.row(i)*b.column(j));});
}

/** @brief Computes the kronecker tensor product
 *  @param a an \f$M \times N\f$ matrix
 *  @param b an \f$P \times Q\f$ matrix
 *  @return An \f$ MP \times NQ \f$ matrix \f$ \textbf{A}\otimes\textbf{B} \f$ of type T such that
 *  \f$ \left(\textbf{AB}\right)_{ij} = a_{\lfloor i/P \rfloor,\lfloor j/Q \rfloor}b_{i\%P,j\%Q} \f$
 *  where \f$ i \% P \f$ is the remainder of \f$ i/P \f$ 
 *
 * Computes the kronecker tensor product of two matrices.
 */
template <typename T, std::size_t M, std::size_t N, 
                      std::size_t P, std::size_t Q>
constexpr matrix<T, M * P, N * Q> kron(const matrix<T, M, N> &a,
                                       const matrix<T, P, Q> &b) {
  return generate<M * P, N * Q>([&a, &b](auto i, auto j) { return a[i / P][j / Q] * b[i % P][j % Q]; });
}

/** @brief computes the matrix inverse 
 *  @param m an \f$ M \times M \f$ matrix of type T
 *  @return The inverse of \f$ \textbf{M} \f$, \f$ \textbf{M}^{-1}\f$ such that 
 *  \f$ \textbf{M}\textbf{M}^{-1} == \textbf{M}^{-1}\textbf{M} == \textbf{I}_{M} \f$
 *
 *  Computes the inverse of a matrix..
 */
template <typename T, std::size_t M>
constexpr matrix<T, M, M> inverse(const matrix<T, M, M> &m) {
  matrix<T, M, M> inv = identity<T, M>;
  auto A = m;

  for (std::size_t k = 0; k < M; ++k) {
    // first normalize the row
    auto scale_A = A[k][k];
    if (scale_A == 0)
      throw "Matrix is not invertible";

    for (std::size_t j = 0; j < M; ++j) {
      inv[k][j] = inv[k][j] / scale_A;
      A[k][j] = A[k][j] / scale_A;
    }
    auto pivot_A = A.row(k);
    auto pivot_I = inv.row(k);


    for (std::size_t i = 0; i < M; ++i) {
      if (i != k) {
        // pivot should be normalized to 1 already
        auto row_to_subtract_A = pivot_A * A[i][k];
        auto row_to_subtract_I = pivot_I * A[i][k];
        for (std::size_t j = 0; j < M; ++j) {
          inv[i][j] = inv[i][j] - row_to_subtract_I[j];
          A[i][j] = A[i][j] - row_to_subtract_A[j];
        }
      }
    }
  }
  return inv;
}

} // namespace cotila

#endif // COTILA_MATRIX_MATH_H_
