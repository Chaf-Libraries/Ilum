#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Context.hpp"
#include "Engine/Subsystem.hpp"

#include "Eventing/Event.hpp"

#include "Graphics/Image/Image.hpp"
#include "Graphics/Image/Sampler.hpp"

#include "RenderGraph/RenderGraphBuilder.hpp"

#include "Loader/ResourceCache.hpp"

namespace Ilum
{
class Renderer : public TSubsystem<Renderer>
{
  public:
	static VkExtent2D RenderTargetSize;

	struct ImageData
	{
		ImageReference image;
		uint32_t       texID;
	};

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

  public:
	Renderer(Context *context = nullptr);

	~Renderer();

	virtual bool onInitialize() override;

	virtual void onPreTick() override;

	virtual void onPostTick() override;

	virtual void onShutdown() override;

	const RenderGraph *getRenderGraph() const;

	ResourceCache &getResourceCache();

	void resetBuilder();

	void rebuild();

	bool isDebug() const;

	void setDebug(bool enable);

	const Sampler &getSampler(SamplerType type) const;

  public:
	std::function<void(RenderGraphBuilder &)> buildRenderGraph = nullptr;

  private:
	void createSamplers();

  private:
	std::function<void(RenderGraphBuilder &)> defaultBuilder;

	RenderGraphBuilder m_rg_builder;

	scope<RenderGraph> m_render_graph = nullptr;

	scope<ResourceCache> m_resource_cache = nullptr;

	std::unordered_map<SamplerType, Sampler> m_samplers;

	bool m_resize = false;

#ifdef _DEBUG
	bool m_debug = true;
#else
	bool m_debug = false;
#endif        // _DEBUG

  public:
	Event<> Event_RenderGraph_Rebuild;
};
}        // namespace Ilum