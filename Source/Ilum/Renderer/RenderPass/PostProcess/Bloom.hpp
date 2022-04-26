#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
class BloomMask : public TRenderPass<BloomMask>
{
  public:
	BloomMask(const std::string &input, const std::string &output);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_input;
	std::string m_output;
};

class BloomDownSample : public TRenderPass<BloomDownSample>
{
  public:
	BloomDownSample(const std::string &input, const std::string &output, uint32_t level);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_input;
	std::string m_output;
	uint32_t    m_level;
};

class BloomUpSample : public TRenderPass<BloomUpSample>
{
  public:
	BloomUpSample(const std::string &low_resolution, const std::string &high_resolution, const std::string &output, uint32_t level);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	uint32_t    m_level;
	std::string m_low_resolution;
	std::string m_high_resolution;
	std::string m_output;
};

class BloomBlend : public TRenderPass<BloomBlend>
{
  public:
	BloomBlend(const std::string &input, const std::string &bloom, const std::string &output);

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_input;
	std::string m_bloom;
	std::string m_output;
};
}        // namespace Ilum::pass