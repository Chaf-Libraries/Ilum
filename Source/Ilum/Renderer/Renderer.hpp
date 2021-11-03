#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Context.hpp"
#include "Engine/Subsystem.hpp"

#include "Eventing/Event.hpp"

#include "Graphics/Image/Image.hpp"
#include "Graphics/Image/Sampler.hpp"

#include "RenderGraph/RenderGraphBuilder.hpp"

#include "Loader/ResourceCache.hpp"

#include "Scene/Component/Camera.hpp"

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

  public:
	Renderer(Context *context = nullptr);

	~Renderer();

	virtual bool onInitialize() override;

	virtual void onPreTick() override;

	virtual void onPostTick() override;

	virtual void onShutdown() override;

	const RenderGraph *getRenderGraph() const;

	RenderGraph *getRenderGraph();

	ResourceCache &getResourceCache();

	void resetBuilder();

	void rebuild();

	bool isDebug() const;

	void setDebug(bool enable);

	bool hasImGui() const;

	void setImGui(bool enable);

	const Sampler &getSampler(SamplerType type) const;

	const VkExtent2D &getRenderTargetExtent() const;

	void resizeRenderTarget(VkExtent2D extent);

	const ImageReference getDefaultTexture() const;

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

	VkExtent2D m_render_target_extent;

	Image m_default_texture;

	bool m_update = false;

	bool m_imgui = true;

	bool m_debug = true;

	uint32_t m_texture_count = 0;

  public:
	struct
	{
		cmpt::Camera camera;

		glm::vec3 position = {0.f, 0.f, 0.f};

		float pitch = 0.f;
		float yaw   = 0.f;

		glm::vec3 front = {1.f, 0.f, 0.f};
		glm::vec3 right = {0.f, 0.f, 1.f};
		glm::vec3 up    = {0.f, 1.f, 0.f};

		glm::mat4 view       = glm::mat4(1.f);
		glm::mat4 projection = glm::perspective(glm::radians(camera.fov), camera.aspect, camera.near, camera.far);

		float speed       = 5.f;
		float sensitivity = 0.5f;
	} Main_Camera;

  public:
	Event<> Event_RenderGraph_Rebuild;
};
}        // namespace Ilum