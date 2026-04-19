# Tensor (Phase 1): get the compile-time tensor working

**Date:** 2026-04-19
**Branch:** `drubinstein/tensor`
**Scope:** Phase 1 only (Approach B — keep nested-aggregate storage, fix helpers). Phase 2 (flat storage, Approach C) is deferred to a separate spec.

## 1. Goal

Make `include/cotila/tensor/` compile cleanly, with every `static_assert` in `test/tensor_test.h` passing and with feature parity to `cotila::matrix` / `cotila::vector` for operators, plus a working `tensordot`. Everything compile-time: no runtime allocation, no runtime dispatch, no exceptions at evaluation time. The test binary succeeds iff the `static_assert`s hold, matching the existing project convention.

## 2. Non-goals (deferred)

- Flat-storage rewrite (Approach C).
- Non-owning tensor views / proxy indexing.
- Broadcasting, reshape, general transpose / axis permutation.

## 3. Current-state diagnosis (what's broken on `drubinstein/tensor`)

Reproduced by `cd test && make`:

1. **Ambiguous `elementwise` overloads** (`tensor/utility.h:20` and `tensor/utility.h:45`). `tensor<T, Dim, Rest...>` with empty `Rest` matches both the base-case and the recursive template; partial ordering does not pick a winner.
2. **Broken recursive call** inside the recursive `elementwise`: `elementwise<F, T, Tensors..., Rest...>(...)` mixes a type pack (`Tensors...`) with a non-type pack (`Rest...`) in the explicit template-argument list.
3. **`operator[]` returns by value** in both `tensor<T, Dim, Rest...>` and `tensor<T, Dim>`. Writes like `op_applied[i] = ...` inside `elementwise` assign into a temporary prvalue — the store never lands.
4. **Missing `<algorithm>` include** in `tensor.h`: `std::copy_n` and `std::transform` used in `slice` don't resolve.
5. **Missing operator overloads**: `operator-` in all flavors, and `tensor / scalar` (only `scalar / tensor` and `tensor / tensor` exist today).
6. **Broken `generate`**: the non-recursive overload declares an unused template parameter `T`; one overload doesn't return; the recursive call passes `T` incorrectly.
7. **`tensordot` stub** (`tensor/math.h`) is empty and takes a runtime `std::size_t axis` parameter — not compile-time.

## 4. Architecture

Storage is unchanged: `tensor<T, Dim, Rest...>` is an aggregate holding `tensor<T, Rest...> array[Dim]`, with base-case specialization `tensor<T, Dim>` holding `T array[Dim]`. All aggregate initialization in `tensor_test.h` continues to work unchanged. Deduction guides stay as they are.

What changes is the helper functions. Each of `elementwise` and `generate` collapses to **one** function template parameterized on `tensor<T, Dim, Rest...>`, branching on `sizeof...(Rest) == 0` with `if constexpr`. That removes the ambiguous-overload class of bug at the template level: there is only one template, so there is no ambiguity to resolve.

Operators delegate to `elementwise`, matching the pattern already used in `cotila::matrix` and `cotila::vector`.

## 5. Components

### 5.1 `tensor` struct (`include/cotila/tensor/tensor.h`)

- Add `#include <algorithm>` for `std::copy_n` / `std::transform` used in `slice`.
- Replace the `// TODO` comment with `static_assert(Dimension != 0 && ((Shape != 0) && ...), "tensor dimensions must be positive")` in the recursive variant, and `static_assert(Dimension != 0, ...)` in the base case.
- Fix `operator[]`:
  - Recursive: `constexpr tensor<T, Shape...>& operator[](std::size_t i) noexcept` and `constexpr const tensor<T, Shape...>& operator[](std::size_t i) const noexcept`.
  - Base: `constexpr T& operator[](std::size_t i) noexcept` and `constexpr const T& operator[](std::size_t i) const noexcept`.
- `slice` member functions stay as they are aside from resolving via `<algorithm>`.
- Aggregate initialization, deduction guides, iterators unchanged.

### 5.2 Unified `elementwise` (`include/cotila/tensor/utility.h`)

Single template; `if constexpr` selects base vs recursive behavior:

```cpp
template <typename F, typename T, std::size_t Dim, std::size_t... Rest,
          typename... Tensors,
          typename U = std::invoke_result_t<F, T, typename Tensors::value_type...>>
constexpr tensor<U, Dim, Rest...>
elementwise(F f, const tensor<T, Dim, Rest...>& t, const Tensors&... tensors) {
  tensor<U, Dim, Rest...> out{};
  for (std::size_t i = 0; i < Dim; ++i) {
    if constexpr (sizeof...(Rest) == 0)
      out[i] = std::apply(f, std::forward_as_tuple(t[i], tensors[i]...));
    else
      out[i] = elementwise(f, t[i], tensors[i]...);
  }
  return out;
}
```

The recursive call uses deduction — no explicit template-argument list. `Tensors` is the pack of co-tensors (same shape as `t`); `value_type` on each is the scalar type at the leaves (already defined on both tensor variants).

### 5.3 Unified `generate` (`include/cotila/tensor/utility.h`)

```cpp
template <typename T, std::size_t Dim, std::size_t... Rest,
          typename F, typename... Idx>
constexpr tensor<T, Dim, Rest...> generate(F f, Idx... prior) {
  tensor<T, Dim, Rest...> out{};
  for (std::size_t i = 0; i < Dim; ++i) {
    if constexpr (sizeof...(Rest) == 0)
      out[i] = f(prior..., i);
    else
      out[i] = generate<T, Rest...>(f, prior..., i);
  }
  return out;
}
```

Callable as `generate<double, 3, 3, 1>([](auto... idx){ return /* ... */; })`.

### 5.4 Multi-axis access helpers (`include/cotila/tensor/utility.h`)

Used by `tensordot`. Both are `constexpr`. Each is a **single** template that matches the primary `tensor<T, Dim, Rest...>`; branching on `sizeof...(Rest) == 0` uses `if constexpr`, same pattern as the unified `elementwise`. This avoids the base-vs-recursive overload-ambiguity class of bug. Index count is checked at the call site with `static_assert`.

```cpp
template <typename T, std::size_t Dim, std::size_t... Rest, typename... Is>
constexpr decltype(auto) at(const tensor<T, Dim, Rest...>& t,
                            std::size_t i, Is... rest) {
  static_assert(sizeof...(Is) == sizeof...(Rest), "wrong number of indices");
  if constexpr (sizeof...(Rest) == 0)
    return t[i];                  // const T&
  else
    return at(t[i], rest...);
}

template <typename T, std::size_t Dim, std::size_t... Rest, typename... Is>
constexpr void set(tensor<T, Dim, Rest...>& t, T v,
                   std::size_t i, Is... rest) {
  static_assert(sizeof...(Is) == sizeof...(Rest), "wrong number of indices");
  if constexpr (sizeof...(Rest) == 0)
    t[i] = v;
  else
    set(t[i], v, rest...);
}
```

### 5.5 Operators (`include/cotila/tensor/operators.h`)

Additions and fixes. All delegate to `elementwise`.

- `tensor - scalar`, `scalar - tensor`, `tensor - tensor` (new).
- `tensor / scalar` (new — only `scalar / tensor` and `tensor / tensor` exist).
- Existing `+`, `*`, `==`, `!=` continue to work once `elementwise` is unified.

Fix the duplicate `operator==` overloads (currently one for the recursive case, one for the base case) by collapsing to a single template, matching the unified-helpers pattern.

### 5.6 `tensordot` (`include/cotila/tensor/math.h`)

Signature:

```cpp
template <std::size_t AxisA, std::size_t AxisB,
          typename T, std::size_t... As, std::size_t... Bs>
constexpr auto tensordot(const tensor<T, As...>& a,
                         const tensor<T, Bs...>& b);
```

- `AxisA`, `AxisB` are template parameters → compile-time.
- `static_assert(AxisA < sizeof...(As) && AxisB < sizeof...(Bs), "axis out of range")`.
- `static_assert` that contracted-axis sizes match: `std::array<std::size_t, sizeof...(As)>{As...}[AxisA] == std::array<std::size_t, sizeof...(Bs)>{Bs...}[AxisB]`.
- Result shape = `As` with position `AxisA` removed, concatenated with `Bs` with position `AxisB` removed. Computed via `std::index_sequence` helpers: `shape_drop<I, S...>` and `shape_cat<IS1, IS2>`; a final `apply_seq` turns the `index_sequence` back into a `tensor<T, ...>` type.
- Body (all constexpr):

```cpp
// pseudocode
for each multi-index ia over (As minus AxisA):
  for each multi-index ib over (Bs minus AxisB):
    T acc = T{};
    for (std::size_t c = 0; c < ContractSize; ++c) {
      auto a_idx = insert_at<AxisA>(ia, c);  // std::array<std::size_t, |As|>
      auto b_idx = insert_at<AxisB>(ib, c);  // std::array<std::size_t, |Bs|>
      acc += at(a, a_idx[0], a_idx[1], ...) * at(b, b_idx[0], b_idx[1], ...);
    }
    set(result, acc, ia[0], ..., ib[0], ...);
```

Iteration over multi-indices is implemented with a small constexpr "odometer" — a `std::array<std::size_t, Rank>` that ticks through the shape. Fanning `std::array` values out as parameter-pack arguments to `at` / `set` uses `std::apply` with an intermediate tuple.

**Rank-0 edge case.** `tensor<T, 0>` is disallowed (the static_assert above makes it explicit). If `AxisA` is the only axis of `a` and `AxisB` is the only axis of `b` (both rank-1 inputs), the result has rank 0. We short-circuit this case to return `T` directly, so `tensordot<0, 0>(u, v)` on two `tensor<T, N>`s is a scalar. No special rank-0 tensor type needed.

### 5.7 `cotila.h`

Add `#include <cotila/tensor/math.h>` if not already present (it currently isn't — only `tensor.h`, `utility.h`, `operators.h` are exported).

## 6. Data flow

- Construction: aggregate initialization of nested arrays — unchanged.
- Indexing: `operator[]` returns a reference; `at(t, i1, i2, ...)` recurses through references. Both `constexpr`.
- Element-wise ops / operators: `operator±*/` → `elementwise(lambda, …)` → constexpr `for` over `Dim`, recurses into `elementwise` on each sub-tensor until `sizeof...(Rest) == 0`.
- Slice: member function reads from `this->array`, writes to a local `sliced.array` via `std::copy_n` / `std::transform` (both constexpr in C++20), returns by value.
- `tensordot`: result shape computed purely at type level via `index_sequence` TMP; body is a constexpr nested loop over an "odometer" multi-index.

## 7. Error handling

All diagnostics are `static_assert`. No runtime checks, no exceptions at evaluation time.

- Zero-length dimensions rejected with `static_assert` in both `tensor` variants.
- `tensordot` axis bounds and contract-size equality checked with `static_assert`.
- Shape mismatch in `elementwise` / operators is surfaced naturally: `tensor<T, 3, 3>` and `tensor<T, 3, 4>` are different types, so overload resolution just fails.

## 8. Testing

All additions in `test/tensor_test.h` as `static_assert`s (the project's convention — the test binary only has to compile).

Covered cases:

- Existing (fixed): `t1 + 1.0`, `t3 * 3.0`, `t31 / 3.0`, `t33 - 3.0`, 3-D slice of `t331`.
- Symmetric operands: `1.0 + t1`, `3.0 * t3`, `3.0 / t31`, `3.0 - t33`.
- Element-wise tensor op tensor: at rank 2 and rank 3, for `+`, `-`, `*`, `/`.
- Equality and inequality of tensors.
- `generate<double, 2, 2>([](auto i, auto j){ return double(i + j); })` produces expected values.
- `tensordot<1, 0>` on `tensor<T, 2, 3>` and `tensor<T, 3, 4>` matches a known `2x4` product.
- `tensordot<0, 0>` on two `tensor<T, 3>`s returns the expected scalar (rank-0 short-circuit).
- `tensordot<2, 0>` on `tensor<T, 2, 3, 4>` and `tensor<T, 4, 5>` produces a `tensor<T, 2, 3, 5>` with at least one spot-checked entry.

## 9. File-level change summary

- `include/cotila/tensor/tensor.h` — add `<algorithm>`, fix `operator[]` to return references, replace TODO assert with real `static_assert`, keep `slice` body unchanged otherwise.
- `include/cotila/tensor/utility.h` — replace both `elementwise` overloads with one unified template; replace `generate` overloads with one unified template; add `at` and `set` multi-axis helpers.
- `include/cotila/tensor/operators.h` — collapse duplicate `operator==`; add `operator-` (three flavors); add `tensor / scalar`; keep existing semantics.
- `include/cotila/tensor/math.h` — implement `tensordot<AxisA, AxisB>`, plus `shape_drop`, `shape_cat`, rank-0 short-circuit.
- `include/cotila/cotila.h` — `#include <cotila/tensor/math.h>`.
- `test/tensor_test.h` — additional `static_assert` coverage as listed in §8.

## 10. Phasing

Implementation order (each step compiles cleanly before moving on):

1. `tensor.h` fixes (`<algorithm>`, reference-returning `operator[]`, positive-dimension `static_assert`).
2. Unified `elementwise` in `utility.h`. Confirm existing ops compile.
3. Fix / add operators (`-` family, `tensor / scalar`, unify `==`). Confirm existing `static_assert`s pass.
4. Unified `generate` + `at` / `set` helpers. Confirm `generate` test case passes.
5. `tensordot` (shape TMP first, then the compute body, then the rank-0 short-circuit). Confirm `tensordot` test cases pass.
6. Add the new `static_assert`s as each piece lands (don't leave them all to the end).
