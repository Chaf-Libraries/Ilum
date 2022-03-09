#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class CopyLastFrame : public TRenderPass<CopyLastFrame>
{
  public:
	CopyLastFrame(const std::string &last_frame_name);

	~CopyLastFrame() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

  private:
	std::string m_last_frame_name;
};
}        // namespace Ilum::pass