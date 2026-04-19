/** @file
 *  @brief Mathematical operations on tensors.
 */
#ifndef COTILA_TENSOR_MATH_H_
#define COTILA_TENSOR_MATH_H_

#include <cotila/tensor/tensor.h>
#include <cotila/tensor/utility.h>

namespace cotila {

    template<typename T, std::size_t... N, std::size_t... M>
    constexpr auto tensordot(const tensor<T, N...> &a, const tensor<T, M...> &b, const std::size_t axis = 0, const std::size_t axis2 = 0) {
        static_assert(sizeof...(N) == sizeof...(M), "Tensors must have the same number of dimensions");
        
    }

} // namespace cotila

#endif // COTILA_TENSOR_MATH_H_