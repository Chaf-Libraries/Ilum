#include "RHIContext.hpp"

namespace Ilum::RHI
{
//std::function<std::shared_ptr<Ilum::RHI::RHIContext>(void)> Ilum::RHI::RHIContext::CreateFunc =
//    []() -> std::shared_ptr<Ilum::RHI::RHIContext> { return nullptr; };

std::shared_ptr<RHIContext> RHIContext::Create()
{
	return CreateFunc();
}

GraphicsBackend RHIContext::GetGraphicsBackend() const
{
	return m_backend;
}
}        // namespace Ilum::RHI