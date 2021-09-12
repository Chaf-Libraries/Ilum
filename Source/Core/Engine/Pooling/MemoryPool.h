#pragma once

#include <memory_resource>

namespace Ilum
{
class MemoryPool
{
  public:
	MemoryPool() = default;

	~MemoryPool();

	MemoryPool(const MemoryPool &) = delete;

	MemoryPool &operator=(const MemoryPool &) = delete;

	MemoryPool(MemoryPool &&) = delete;

	MemoryPool &operator=(MemoryPool &&) = delete;

	template <typename T, typename... Args>
	T *create(Args &&...args);

	template <typename T>
	void destroy(T *&elem);

  private:
	std::pmr::synchronized_pool_resource m_pool;
};
}        // namespace Ilum

#include "MemoryPool.inl"