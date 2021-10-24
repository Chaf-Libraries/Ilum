#pragma once

#include "Utils/PCH.hpp"

#include "Graphics/Pipeline/PipelineState.hpp"
#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class ImGuiPass : public TRenderPass<ImGuiPass>
{
  public:
	ImGuiPass(const std::string &output_name, AttachmentState state = AttachmentState::Load_Color);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

  private:
	std::string     m_output;
	AttachmentState m_attachment_state;
};
}        // namespace Ilum