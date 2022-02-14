#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Context.hpp"
#include "Engine/Subsystem.hpp"

#include <Core/Event.hpp>

#include <Graphics/Resource/Image.hpp>
#include <Graphics/Resource/Sampler.hpp>

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

	const Graphics::Sampler &getSampler(SamplerType type) const;

	const VkExtent2D &getRenderTargetExtent() const;

	void resizeRenderTarget(VkExtent2D extent);

	const Graphics::ImageReference getDefaultTexture() const;

	bool hasMainCamera();

	void update();

  public:
	std::function<void(RenderGraphBuilder &)> buildRenderGraph = nullptr;

  private:
	void createSamplers();

	void updateImages();

  private:
	std::function<void(RenderGraphBuilder &)> DeferredRendering;

	RenderGraphBuilder m_rg_builder;

	scope<RenderGraph> m_render_graph = nullptr;

	scope<ResourceCache> m_resource_cache = nullptr;

	std::unordered_map<SamplerType, Graphics::Sampler> m_samplers;

	VkExtent2D m_render_target_extent;

	Graphics::Image m_default_texture;

	bool m_update = false;

	bool m_imgui = true;

	std::array<bool, 3> m_recorded = {false};

	uint32_t m_texture_count = 0;

  public:
	Entity Main_Camera;

	RenderStats Render_Stats;

	RenderBuffer Render_Buffer;

	RenderMode Render_Mode = RenderMode::Polygon;

	struct
	{
		// E(mu)
		Graphics::Image kulla_conty_energy = Graphics::Image(Graphics::RenderContext::GetDevice());
		// Eavg
		Graphics::Image kulla_conty_energy_average = Graphics::Image(Graphics::RenderContext::GetDevice());
	}PreCompute;

	struct
	{
		float exposure = 4.5f;
		float gamma    = 2.2f;
	} Color_Correction;

	struct
	{
		float    threshold = 0.75f;
		float    scale     = 3.f;
		float    strength  = 0.13f;
		uint32_t enable    = 0;
	} Bloom;

	struct
	{
		bool enable = true;
		glm::vec2 current_jitter = glm::vec2(0.f);
		glm::vec2 prev_jitter    = glm::vec2(0.f);
		glm::vec2 feedback       = glm::vec2(1.f, 1.f);
	}TAA;

	struct
	{
		scope<Graphics::Image> depth_buffer = nullptr;
		scope<Graphics::Image> hiz_buffer   = nullptr;
		scope<Graphics::Image> last_result  = nullptr;
	} Last_Frame;

	struct
	{
		uint32_t frustum_culling   = 1;
		uint32_t backface_culling  = 1;
		uint32_t occulsion_culling = 0;
	} Culling;

	enum class EnvLightType
	{
		None,
		HDR,
		Cubemap,
		// TODO: Atmospheric
	};

	struct
	{
		EnvLightType type = EnvLightType::None;
		std::string  filename;
		bool         update = false;
	} EnvLight;

  public:
	Core::Event<> Event_RenderGraph_Rebuild;
};
}        // namespace Ilum