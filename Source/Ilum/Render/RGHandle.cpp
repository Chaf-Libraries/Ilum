#include "RGHandle.hpp"

namespace Ilum
{
RGHandle::RGHandle() :
    m_index(CURRENT_ID++)
{
}

RGHandle::RGHandle(uint32_t index):
    m_index(index)
{
	if (CURRENT_ID <= index && index != INVALID_ID)
	{
		CURRENT_ID = index + 1;
	}
}

RGHandle::operator uint32_t() const
{
	return m_index;
}

void RGHandle::Invalidate()
{
	m_index = INVALID_ID;
}

bool RGHandle::IsInvalid() const
{
	return m_index == INVALID_ID;
}

bool RGHandle::operator<(const RGHandle &rhs)
{
	return m_index < rhs.m_index;
}

bool RGHandle::operator==(const RGHandle &rhs)
{
	return m_index == rhs.m_index;
}

void RGHandle::Reset()
{
	CURRENT_ID = 0;
}

void RGHandle::SetCurrent(uint32_t id)
{
	CURRENT_ID = id;
}
}