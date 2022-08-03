#include "RHICommand.hpp"

#include "Backend/Vulkan/Command.hpp"

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

std::unique_ptr<RHICommand> RHICommand::Create(RHIDevice *device, uint32_t family_index, RHIQueueFamily family)
{
	return std::make_unique<Vulkan::Command>(device, family_index, family);
}
}        // namespace Ilum