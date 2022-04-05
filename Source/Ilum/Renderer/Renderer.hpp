#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Context.hpp"
#include "Engine/Subsystem.hpp"

#include "Eventing/Event.hpp"

#include "Graphics/Image/Image.hpp"
#include "Graphics/Image/Sampler.hpp"

#include "Scene/Entity.hpp"

#include "RenderGraph/RenderGraphBuilder.hpp"

#include "Loader/ResourceCache.hpp"

#include "RenderData.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
class Renderer : public TSubsystem<Renderer>
{
  public:
	enum class SamplerType
	{
		Compare_Depth,
		Point_Clamp,
		Point_Wrap,
		Bilinear_Clamp,
		Bilinear_Wrap,
		Trilinear_Clamp,
		Trilinear_Wrap,
		Anisptropic_Clamp,
		Anisptropic_Wrap
	};

	enum class RenderMode
	{
		Polygon,
		WireFrame,
		PointCloud
	};

  public:
	Renderer(Context *context = nullptr);

	~Renderer();

	virtual bool onInitialize() override;

	virtual void onPreTick() override;

	virtual void onTick(float delta_time) override;

	virtual void onPostTick() override;

	virtual void onShutdown() override;

	const RenderGraph *getRenderGraph() const;

	RenderGraph *getRenderGraph();

	ResourceCache &getResourceCache();

	void rebuild();

	bool hasImGui() const;

	void setImGui(bool enable);

	const Sampler &getSampler(SamplerType type) const;

	const VkExtent2D &getViewportExtent() const;

	const VkExtent2D &getRenderTargetExtent() const;

	void resizeViewport(VkExtent2D extent);

	void resizeRenderTarget(VkExtent2D extent);

	const ImageReference getDefaultTexture() const;

	bool hasMainCamera();

	void update();

  public:
	std::function<void(RenderGraphBuilder &)> buildRenderGraph = nullptr;

  private:
	void createSamplers();

  private:
	std::function<void(RenderGraphBuilder &)> DeferredRendering;

	RenderGraphBuilder m_rg_builder;

	scope<RenderGraph> m_render_graph = nullptr;

	scope<ResourceCache> m_resource_cache = nullptr;

	std::unordered_map<SamplerType, Sampler> m_samplers;

	VkExtent2D m_viewport_extent;

	VkExtent2D m_render_target_extent;

	Image m_default_texture;

	bool m_update = false;

	bool m_imgui = true;

	std::array<bool, 3> m_recorded = {false};

	uint32_t m_texture_count = 0;

  public:
	Entity Main_Camera;

	RenderStats Render_Stats;

	RenderBuffer Render_Buffer;

	RenderMode Render_Mode = RenderMode::Polygon;

  public:
	Event<> Event_RenderGraph_Rebuild;
};
}        // namespace Ilum