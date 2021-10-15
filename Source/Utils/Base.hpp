#pragma once

#include <memory>

#ifdef SHARED_LINK
#	define ILUM_IMPORT __declspec(dllimport)
#	define ILUM_EXPORT __declspec(dllexport)
#else
#	define ILUM_IMPORT
#	define ILUM_EXPORT
#endif        // SHARED_LINK

#define DISABLE_WARNINGS()\
		__pragma(warning(push, 0))

#define ENABLE_WARNINGS()\
		__pragma(warning(pop))

#define BIT(x) (1 << x)

#define VAR_NAME(var) (#var)

#define ENUMCLASS_RELOAD_INLINE(_Ty)\
_Ty operator|(const _Ty &lhs, const _Ty &rhs)\
{\
	return (_Ty)((uint32_t) lhs | (uint32_t) rhs);\
}\
_Ty operator&(const _Ty &lhs, const _Ty &rhs)\
{\
	return (_Ty)((uint32_t) lhs & (uint32_t) rhs);\
}\
bool operator==(const _Ty &lhs, const _Ty &rhs)\
{\
	return (uint32_t) lhs == (uint32_t) rhs;\
}\
bool operator!=(const _Ty &lhs, const _Ty &rhs)\
{\
	return (uint32_t) lhs != (uint32_t) rhs;\
}

namespace Ilum
{
template <typename T>
using scope = std::unique_ptr<T>;
template <typename T, typename... Args>
scope<T> createScope(Args &&...args)
{
	return std::make_unique<T>(std::forward<Args>(args)...);
}

template <typename T>
using ref = std::shared_ptr<T>;
template <typename T, typename... Args>
ref<T> createRef(Args &&...args)
{
	return std::make_shared<T>(std::forward<Args>(args)...);
}
}        // namespace Ilum