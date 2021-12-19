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
#include "RenderQueue.hpp"

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

	enum class BufferType
	{
		//MainCamera,
		//Vertex,
		//Index,
		/*Material,*/
		//Transform,
		//BoundingBox,
		//Meshlet,
		//IndirectCommand
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

	const BufferReference getBuffer(BufferType type) const;

	const VkExtent2D &getRenderTargetExtent() const;

	void resizeRenderTarget(VkExtent2D extent);

	const ImageReference getDefaultTexture() const;

	bool hasMainCamera();

	void update();

  public:
	std::function<void(RenderGraphBuilder &)> buildRenderGraph = nullptr;

  private:
	void createSamplers();

	void createBuffers();

	void updateBuffers();

	void updateInstanceBuffer();

	void updateImages();

  private:
	std::function<void(RenderGraphBuilder &)> DeferredRendering;

	RenderGraphBuilder m_rg_builder;

	scope<RenderGraph> m_render_graph = nullptr;

	scope<ResourceCache> m_resource_cache = nullptr;

	std::unordered_map<SamplerType, Sampler> m_samplers;
	std::unordered_map<BufferType, Buffer>   m_buffers;

	VkExtent2D m_render_target_extent;

	Image m_default_texture;

	bool m_update = false;

	bool m_imgui = true;

	uint32_t m_texture_count = 0;

  public:
	//Camera Main_Camera_;

	Entity Main_Camera;

	RenderQueue Render_Queue;

	RenderStats Render_Stats;

	RenderBuffer Render_Buffer;

	//uint32_t Static_Instance_Count = 0;
	//uint32_t Instance_Count        = 0;
	//uint32_t Meshlet_Count  = 0;
	//uint32_t Indices_Count   = 0;
	//uint32_t Meshlet_Visible = 0;
	//uint32_t Instance_Visible = 0;

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
		scope<Image> depth_buffer;
		scope<Image> hiz_buffer;
	} Last_Frame;

	struct
	{
		uint32_t frustum_culling   = 1;
		uint32_t backface_culling  = 1;
		uint32_t occulsion_culling = 0;
	} Culling;

  public:
	Event<> Event_RenderGraph_Rebuild;
};
}        // namespace Ilum