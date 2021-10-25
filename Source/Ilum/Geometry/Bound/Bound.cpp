#include "Bound.hpp"
#include "AABB.hpp"

namespace Ilum
{
Bound::Bound()
{
	m_ptr = createScope<AABB>();
}

Bound *Bound::get() const
{
	if (!m_ptr)
	{
		return nullptr;
	}

	return m_ptr.get();
}

void Bound::set(Type type)
{
	if (m_type != type)
	{
		switch (type)
		{
			case Ilum::Bound::Type::AABB:
				m_ptr = createScope<AABB>();
			default:
				break;
		}
		m_type = type;
	}
}
}        // namespace Ilum