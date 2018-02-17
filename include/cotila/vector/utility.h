#ifndef COTILA_VECTOR_UTILITY_H_
#define COTILA_VECTOR_UTILITY_H_

#include <cotila/vector/vector.h>
#include <tuple>

namespace cotila {

template <
    typename F, typename T, typename... Vectors,
    typename U = std::invoke_result_t<F, T, typename Vectors::value_type...>,
    std::size_t N =
        detail::all_same_value<std::size_t, Vectors::size...>::value>
constexpr vector<U, N> elementwise(F f, const vector<T, N> &v,
                                   const Vectors &... vectors) {
  std::array<U, N> op_applied = {};
  for (std::size_t i = 0; i < N; ++i)
    op_applied[i] = std::apply(f, std::forward_as_tuple(v[i], vectors[i]...));
  return op_applied;
}

template <typename T, std::size_t N, typename F, typename U>
constexpr U accumulate(const vector<T, N> &v, U init, F &&f) {
  U r = init;
  for (std::size_t i = 0; i < vector<T, N>::size; ++i)
    r = std::apply(std::forward<F>(f), std::forward_as_tuple(r, v[i]));
  return r;
}

template <typename T, typename U, std::size_t N>
constexpr vector<T, N> cast(const vector<U, N> &v) {
  return elementwise([](const U u) { return static_cast<T>(u); }, v);
}

template <std::size_t N, typename T>
constexpr vector<T, N> iota(T value = T()) {
  std::array<T, N> seq = {};
  for (auto &x : seq)
    x = value++;
  return seq;
}

template <std::size_t N, typename T>
constexpr vector<T, N> linspace(T min, T max) {
  return ((max - min) / (N - 1)) * iota<N, T>() + min;
}

template <std::size_t N, typename T> constexpr vector<T, N> fill(T value) {
  std::array<T, N> filled = {};
  for (auto &x : filled)
    x = value;
  return filled;
}

template <std::size_t N, typename F> constexpr decltype(auto) generate(F &&f) {
  return elementwise(f, iota<N, std::size_t>());
}

template <typename T, typename U, std::size_t N>
constexpr vector<std::tuple<T, U>, N> zip(vector<T, N> vec1,
                                          vector<U, N> vec2) {
  std::array<std::tuple<T, U>, N> zipped = {};
  for (std::size_t i = 0; i < N; ++i) {
    std::get<0>(zipped[i]) = vec1[i];
    std::get<1>(zipped[i]) = vec2[i];
  }
  return zipped;
}

} // namespace cotila

#endif // COTILA_VECTOR_UTILITY_H_
