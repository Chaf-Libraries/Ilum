#pragma once

#include <RenderGraph/ResourceNode.hpp>

namespace Ilum::Render
{
class RenderGraph;

class BindlessTextureNode : public IResourceNode
{
  public:
	BindlessTextureNode(RenderGraph& render_graph);
	~BindlessTextureNode() = default;

	virtual void OnImGui() override;
	virtual void OnImNode() override;
	virtual void OnUpdate() override;

  protected:
	virtual bool _ReadBy(IPassNode *pass, int32_t pin) override;
	virtual bool _WriteBy(IPassNode *pass, int32_t pin) override;

  private:
	std::vector<Graphics::ImageReference> m_images;
};
}        // namespace Ilum::Render