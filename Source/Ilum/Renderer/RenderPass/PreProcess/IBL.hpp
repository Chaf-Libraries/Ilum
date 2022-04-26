#pragma once

#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

#include <glm/glm.hpp>

namespace Ilum::pass
{
class EquirectangularToCubemap : public TRenderPass<EquirectangularToCubemap>
{
  public:
	EquirectangularToCubemap() = default;

	~EquirectangularToCubemap();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::string m_filename = "";

	struct
	{
		glm::mat4 inverse_view_projection;
		uint32_t  tex_idx;
	} m_push_data;

	std::vector<VkFramebuffer> m_framebuffers;
};

class BRDFPreIntegrate : public TRenderPass<BRDFPreIntegrate>
{
  public:
	BRDFPreIntegrate() = default;

	~BRDFPreIntegrate() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	bool m_finish = true;
};

class CubemapSHProjection : public TRenderPass<CubemapSHProjection>
{
  public:
	CubemapSHProjection() = default;

	~CubemapSHProjection() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	int32_t m_face_id = 0;
};

class CubemapSHAdd : public TRenderPass<CubemapSHAdd>
{
  public:
	CubemapSHAdd() = default;

	~CubemapSHAdd() = default;

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	int32_t m_face_id = 0;
};

class CubemapPrefilter : public TRenderPass<CubemapPrefilter>
{
  public:
	CubemapPrefilter() = default;

	~CubemapPrefilter();

	virtual void setupPipeline(PipelineState &state) override;

	virtual void resolveResources(ResolveState &resolve) override;

	virtual void render(RenderPassState &state) override;

	virtual void onImGui() override;

  private:
	std::vector<VkDescriptorSet> m_descriptor_sets;
	std::vector<VkImageView>     m_views;
	const uint32_t               m_mip_levels    = 5;
	int32_t                      m_current_level = 0;

	struct
	{
		VkExtent2D mip_extent = {};
		float      roughness  = 0.f;
	} m_push_data;
};
}        // namespace Ilum::pass