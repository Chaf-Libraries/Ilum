#pragma once

#include "RenderGraph/PassNode.hpp"

namespace Ilum::Render
{
class GeometryPass : public IPassNode
{
  public:
	GeometryPass(RenderGraph &render_graph);
	~GeometryPass() = default;

	virtual void OnExecute(Graphics::CommandBuffer &cmd_buffer) override;
};
}        // namespace Ilum::Render