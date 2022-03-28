#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class CopyFrame : public TRenderPass<CopyFrame>
{
  public:
	CopyFrame(const std::string &from, const std::string& to);

	~CopyFrame() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

  private:
	std::string m_from;
	std::string m_to;
};
}        // namespace Ilum::pass