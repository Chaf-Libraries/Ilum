#pragma once

#include "PassNode.hpp"
#include "ResourceNode.hpp"

namespace Ilum::Render
{
class RenderGraph
{
  public:
	RenderGraph() = default;

	template <typename T>
	void AddResourceNode()
	{
		std::unique_ptr<IResourceNode> ptr = std::make_unique<T>(*this);
		m_render_nodes[ptr->GetUUID()]     = ptr.get();
		m_resource_nodes[ptr->GetUUID()]= std::move(ptr);
	}

	template <typename T>
	void AddPassNode()
	{
		std::unique_ptr<IPassNode> ptr = std::make_unique<T>(*this);
		m_render_nodes[ptr->GetUUID()] = ptr.get();
		m_pass_nodes[ptr->GetUUID()] = std::move(ptr);
	}

	void RegisterPin(int32_t pin, int32_t node);
	void UnRegisterPin(int32_t pin);

	void OnImGui();

  private:
	std::map<int32_t, std::unique_ptr<IResourceNode>> m_resource_nodes;
	std::map<int32_t, std::unique_ptr<IPassNode>>     m_pass_nodes;
	std::map<int32_t, RenderNode *>                   m_render_nodes;
	std::map<int32_t, int32_t>                        m_pin_query;        //Pin -> Node
	std::vector<std::pair<int32_t, int32_t>>          m_links;            // Pin->Pin
};
}        // namespace Ilum::Render