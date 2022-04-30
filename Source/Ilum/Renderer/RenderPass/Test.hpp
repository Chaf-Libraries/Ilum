#pragma once

#include "Utils/PCH.hpp"

#include "Graphics/Pipeline/PipelineState.hpp"
#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class TestPass : public TRenderPass<TestPass>
{
  public:
	TestPass();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	Buffer buffer1;
	Buffer buffer2;
	Buffer buffer3;
	Buffer address_buffer;
	int buffer_index = 0;
	struct
	{
		uint32_t color_index;
		uint64_t buffer_address;
	}m_push_constant;
};
}        // namespace Ilum::pass