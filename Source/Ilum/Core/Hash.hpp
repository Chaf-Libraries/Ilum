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

template <typename T>
inline void HashCombine(size_t &seed, const std::vector<T> &v)
{
	for (auto& data : v)
	{
		HashCombine(seed, data);
	}
}

template <class T1, class... T2>
inline void HashCombine(size_t &seed, const T1 &v1, const T2 &...v2)
{
	HashCombine(seed, v1);
	HashCombine(seed, v2...);
}
}        // namespace Ilum