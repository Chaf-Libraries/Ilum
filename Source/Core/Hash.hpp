#pragma once

namespace Ilum::Core
{
template <typename T>
constexpr void HashCombine(size_t &seed, const T &t)
{
	std::hash<T> hasher;
	seed ^= hasher(t) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}        // namespace Ilum