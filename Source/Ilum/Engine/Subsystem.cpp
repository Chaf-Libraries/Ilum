#include "Subsystem.hpp"

namespace Ilum
{
ISubsystem::ISubsystem(Context *context) :
    m_context(context)
{
	s_enable = true;
}

bool ISubsystem::onInitialize()
{
	return true;
}

void ISubsystem::onPreTick()
{
}

void ISubsystem::onTick(float delta_time)
{
}

void ISubsystem::onPostTick()
{
}

void ISubsystem::onShutdown()
{
}
}        // namespace Ilum