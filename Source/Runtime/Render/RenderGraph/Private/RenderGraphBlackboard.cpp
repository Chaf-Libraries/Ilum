#include "RenderGraphBlackboard.hpp"

namespace Ilum
{
std::shared_ptr<void> &RenderGraphBlackboard::Add(std::type_index type, std::shared_ptr<void> &&ptr)
{
	{
		std::lock_guard<std::mutex> lock(m_mutex);

		if (m_data.find(type) == m_data.end())
		{
			m_data.emplace(type, std::move(ptr));
		}
	}

	return m_data.at(type);
}

bool RenderGraphBlackboard::Has(std::type_index type)
{
	return m_data.find(type) != m_data.end();
}

std::shared_ptr<void>& RenderGraphBlackboard::Get(std::type_index type)
{
	return m_data.at(type);
}

RenderGraphBlackboard &RenderGraphBlackboard::Erase(std::type_index type)
{
	if (Has(type))
	{
		m_data.erase(type);
	}
	return *this;
}
}        // namespace Ilum