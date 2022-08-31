#include "RHICommand.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Backend/Vulkan/Command.hpp"
#elif defined RHI_BACKEND_DX12
#	include "Backend/DX12/Command.hpp"
#elif defined RHI_BACKEND_CUDA
#	include "Backend/CUDA/Command.hpp"
#endif        // RHI_BACKEND

namespace Ilum
{
RHICommand::RHICommand(RHIDevice *device, RHIQueueFamily family):
    p_device(device), m_family(family)
{
}

CommandState RHICommand::GetState() const
{
	return m_state;
}

void RHICommand::Init()
{
	ASSERT(m_state == CommandState::Available);
	m_state = CommandState::Initial;
}

std::unique_ptr<RHICommand> RHICommand::Create(RHIDevice *device, RHIQueueFamily family)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Command>(device, family);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Command>(device, family);
#elif defined RHI_BACKEND_CUDA
	return std::make_unique<CUDA::Command>(device, family);
#endif        // RHI_BACKEND
}

}        // namespace Ilum