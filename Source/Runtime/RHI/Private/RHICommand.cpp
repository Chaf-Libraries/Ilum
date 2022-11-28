#include "RHICommand.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

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

const std::string RHICommand::GetBackend() const
{
	return p_device->GetBackend();
}

CommandState RHICommand::GetState() const
{
	return m_state;
}

void RHICommand::Init()
{
	assert(m_state == CommandState::Available);
	m_state = CommandState::Initial;
}

std::unique_ptr<RHICommand> RHICommand::Create(RHIDevice *device, RHIQueueFamily family)
{
	return std::unique_ptr<RHICommand>(std::move(PluginManager::GetInstance().Call<RHICommand *>(fmt::format("RHI.{}.dll", device->GetBackend()), "CreateCommand", device, family)));
}

}        // namespace Ilum