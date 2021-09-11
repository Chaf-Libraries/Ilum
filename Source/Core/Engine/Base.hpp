#pragma once

#include <memory>

#ifdef SHARED_LINK
#	define ILUM_IMPORT __declspec(dllimport)
#	define ILUM_EXPORT __declspec(dllexport)
#else
#	define ILUM_IMPORT
#	define ILUM_EXPORT
#endif        // SHARED_LINK

#define BIT(x) (1 << x)

#define VAR_NAME(var) (#var)

namespace Ilum
{
template <typename T>
using scope = std::unique_ptr<T>;
template <typename T, typename... Args>
constexpr scope<T> createScope(Args &&...args)
{
	return std::make_unique<T>(std::forward<Args>(args)...);
}

template <typename T>
using ref = std::shared_ptr<T>;
template <typename T, typename... Args>
constexpr ref<T> createRef(Args &&...args)
{
	return std::make_shared<T>(std::forward<Args>(args)...);
}
}        // namespace Ilum