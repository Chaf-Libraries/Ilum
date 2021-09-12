#include "MemoryPool.h"

namespace Ilum
{
MemoryPool::~MemoryPool()
{
	m_pool.release();
}
}        // namespace Ilum