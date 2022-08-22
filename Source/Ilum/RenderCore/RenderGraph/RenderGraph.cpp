#include "RenderGraph.hpp"

namespace Ilum
{
RGHandle::RGHandle(size_t handle) :
    m_handle(handle)
{
}

size_t RGHandle::operator()()
{
	return m_handle;
}

bool RGHandle::operator<(const RGHandle &rhs) const
{
	return m_handle < rhs.m_handle;
}

size_t RGHandle::GetHandle() const
{
	return m_handle;
}

}        // namespace Ilum