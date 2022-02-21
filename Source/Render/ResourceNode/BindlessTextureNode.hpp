#pragma once

#include <RenderGraph/ResourceNode.hpp>

namespace Ilum::Render
{
class BindlessTextureNode : public IResourceNode
{
  public:
	BindlessTextureNode();
	~BindlessTextureNode() = default;

	virtual void OnImGui()  override;
	virtual void OnImNode() override;
	virtual void OnUpdate() override;
};
}        // namespace Ilum::Render