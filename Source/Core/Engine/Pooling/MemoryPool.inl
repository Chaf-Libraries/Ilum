#include "MemoryPool.h"

namespace Ilum
{
template <typename T, typename... Args>
inline T *MemoryPool::create(Args &&...args)
{
	T *elem = (T *) m_pool.allocate(sizeof(T), alignof(T));
	new (elem) T(std::forward<Args>(args)...);
	return elem;
}

template <typename T>
inline void MemoryPool::destroy(T *&elem)
{
	m_pool.deallocate(elem, sizeof(T), alignof(T));
	elem = nullptr;
}
}        // namespace Ilum