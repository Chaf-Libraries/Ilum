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

template<typename T>
inline size_t Hash(const T& v)
{
	size_t hash = 0;
	HashCombine(hash, v);
	return hash;
}

template <typename T1, typename... T2>
inline size_t Hash(const T1 &v1, const T2 &...v2)
{
	size_t hash = 0;
	HashCombine(hash, v1, v2...);
	return hash;
}
}        // namespace Ilum
