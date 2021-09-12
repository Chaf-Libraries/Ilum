#pragma once

namespace Ilum
{
template <typename T>
constexpr void hash_combine(uint32_t &seed, const T &t)
{
	std::hash<T> hasher;
	seed ^= hasher(t) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}