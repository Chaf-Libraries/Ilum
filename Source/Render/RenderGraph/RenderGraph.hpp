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
	void AddRenderNode(std::unique_ptr<T> &&node)
	{
		m_render_nodes.emplace(node->GetUUID(), std::move(node));
	}

	void RegisterPin(int32_t uuid, PinDesc * pin_desc);
	void UnRegisterPin(int32_t uuid);

	void OnImGui();

  private:
	std::map<int32_t, std::unique_ptr<RenderNode>> m_render_nodes;
	std::map<int32_t, PinDesc *>                      m_pin_desc;
	std::vector<std::pair<int32_t, int32_t>>          m_links;
};
}        // namespace Ilum::Render