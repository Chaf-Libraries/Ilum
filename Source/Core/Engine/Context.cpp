#include "Context.hpp"

namespace Ilum
{
Context::~Context()
{
	for (size_t i = m_subsystems.size() - 1; i > 0; i--)
	{
		m_subsystems[i].ptr.reset();
	}

	m_subsystems.clear();
}

void Context::onInitialize()
{
	std::vector<uint32_t> failed_indices;

	for (uint32_t i = 0; i < static_cast<uint32_t>(m_subsystems.size()); i++)
	{
		if (!m_subsystems[i].ptr->onInitialize())
		{
			failed_indices.push_back(i);
			LOG_ERROR("Failed to initalize subsystem {}", typeid(*m_subsystems[i].ptr).name());
		}
	}

	// Remove all failed subsystems
	for (uint32_t failed_index : failed_indices)
	{
		m_subsystems.erase(m_subsystems.begin() + failed_index);
	}
}

void Context::onPreTick()
{
	for (uint32_t i = 0; i < static_cast<uint32_t>(m_subsystems.size()); i++)
	{
		m_subsystems[i].ptr->onPreTick();
	}
}

void Context::onTick(TickType tick_group, float delta_time)
{
	for (uint32_t i = 0; i < static_cast<uint32_t>(m_subsystems.size()); i++)
	{
		if (m_subsystems[i].tick_group == tick_group)
		{
			m_subsystems[i].ptr->onTick(delta_time);
		}
	}
}

void Context::onPostTick()
{
	for (uint32_t i = 0; i < static_cast<uint32_t>(m_subsystems.size()); i++)
	{
		m_subsystems[i].ptr->onPostTick();
	}
}

void Context::onShutdown()
{
	for (size_t i = static_cast<uint32_t>(m_subsystems.size()) - 1; i > 0; i--)
	{
		m_subsystems[i].ptr->onShutdown();
	}
}
}        // namespace Tools