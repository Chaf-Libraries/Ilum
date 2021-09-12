#include "MemoryPool.hpp"

namespace Ilum
{
MemoryPool::~MemoryPool()
{
	m_pool.release();
}
}        // namespace Ilum