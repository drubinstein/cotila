#include <tuple>

template <class... Ts>
concept all_same =
    sizeof...(Ts) < 2 ||
    std::conjunction_v<
        std::is_same<std::tuple_element_t<0, std::tuple<Ts...>>, Ts>...>;