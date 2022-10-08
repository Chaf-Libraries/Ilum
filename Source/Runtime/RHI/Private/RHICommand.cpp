#include "RHICommand.hpp"
#include "RHIDevice.hpp"

#include "Backend/DX12/Command.hpp"
#include "Backend/Vulkan/Command.hpp"
#ifdef CUDA_ENABLE
#	include "Backend/CUDA/Command.hpp"
#endif        // CUDA_ENABLE

namespace Ilum
{
RHICommand::RHICommand(RHIDevice *device, RHIQueueFamily family) :
    p_device(device), m_family(family)
{
}

RHIQueueFamily RHICommand::GetQueueFamily() const
{
	return m_family;
}

RHIBackend RHICommand::GetBackend() const
{
	return p_device->GetBackend();
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
	switch (device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Command>(device, family);
		case RHIBackend::DX12:
			return std::make_unique<Vulkan::Command>(device, family);
#ifdef CUDA_ENABLE
		case RHIBackend::CUDA:
			return std::make_unique<Vulkan::Command>(device, family);
#endif
		default:
			break;
	}
	return nullptr;
}

}        // namespace Ilum