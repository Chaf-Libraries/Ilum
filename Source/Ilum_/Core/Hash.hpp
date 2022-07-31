#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

namespace Ilum
{
template <class T>
inline void HashCombine(size_t &seed, const T &v)
{
	std::hash<T> hasher;
	glm::detail::hash_combine(seed, hasher(v));
}
}        // namespace Ilum