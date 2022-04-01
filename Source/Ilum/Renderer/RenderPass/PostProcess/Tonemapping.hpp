#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
// Extract bright part for blooming
class Tonemapping : public TRenderPass<Tonemapping>
{
  public:
	Tonemapping(const std::string &from, const std::string& to);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_from;
	std::string m_to;

	struct
	{
		VkExtent2D extent   = {};
		float exposure = 2.f;
		float gamma    = 5.f;
	} m_tonemapping_data;
};
}        // namespace Ilum::pass