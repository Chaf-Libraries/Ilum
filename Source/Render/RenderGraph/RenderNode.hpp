#pragma once

#include <cstdint>
#include <string>

namespace Ilum::Render
{
class RenderGraph;

class RenderNode
{
  public:
	RenderNode(const std::string &name, RenderGraph &render_graph);
	~RenderNode() = default;

	int32_t            GetUUID() const;
	const std::string &GetName() const;

	virtual void OnUpdate() = 0;
	virtual void OnImGui()  = 0;
	virtual void OnImNode() = 0;

  protected:
	int32_t NewUUID();

  protected:
	RenderGraph &  m_render_graph;
	const int32_t m_uuid;
	std::string    m_name;
};
}        // namespace Ilum::Render