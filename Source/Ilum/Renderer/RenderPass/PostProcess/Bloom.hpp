#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class BloomMask : public TRenderPass<BloomMask>
{
  public:
	BloomMask(const std::string &input);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_input;
};

class BloomDownSample : public TRenderPass<BloomDownSample>
{
  public:
	BloomDownSample(uint32_t level);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	uint32_t m_level;
};

class BloomBlur : public TRenderPass<BloomBlur>
{
  public:
	BloomBlur(uint32_t level);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	uint32_t m_level;
};

class BloomUpSample : public TRenderPass<BloomUpSample>
{
  public:
	BloomUpSample(uint32_t level, bool start = false);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	uint32_t m_level;
	bool     m_start = false;
};

class BloomBlend : public TRenderPass<BloomBlend>
{
  public:
	BloomBlend(const std::string &input, const std::string &output);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_input;
	std::string m_output;
};
}        // namespace Ilum::pass