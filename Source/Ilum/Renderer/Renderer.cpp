#include "Renderer.hpp"
#include "RenderGraph/RenderGraph.hpp"

#include "Renderer/RenderPass/ImGuiPass.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "Loader/ImageLoader/Bitmap.hpp"
#include "Loader/ImageLoader/ImageLoader.hpp"

#include "Scene/Component/DirectionalLight.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/PointLight.hpp"
#include "Scene/Component/SpotLight.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include <imgui.h>

namespace Ilum
{
Renderer::Renderer(Context *context) :
    TSubsystem<Renderer>(context)
{
	GraphicsContext::instance()->Swapchain_Rebuild_Event += [this]() { m_update = true; };

	defaultBuilder = [this](RenderGraphBuilder &builder) {

	};

	buildRenderGraph = defaultBuilder;

	m_resource_cache = createScope<ResourceCache>();
	createSamplers();
	createBuffers();
	ImageLoader::loadImage(m_default_texture, Bitmap{{0, 0, 0, 255}, VK_FORMAT_R8G8B8A8_UNORM, 1, 1}, false);
}

Renderer::~Renderer()
{
}

bool Renderer::onInitialize()
{
	m_render_target_extent = GraphicsContext::instance()->getSwapchain().getExtent();

	defaultBuilder(m_rg_builder);

	rebuild();

	return true;
}

void Renderer::onPreTick()
{
	// Flush resource cache
	m_resource_cache->flush();

	// Update uniform buffers
	updateBuffers();

	// Update camera
	Main_Camera.onUpdate();

	// Check out images update
	if (m_texture_count != m_resource_cache->getImages().size())
	{
		m_update        = true;
		m_texture_count = static_cast<uint32_t>(m_resource_cache->getImages().size());
	}

	if (m_update)
	{
		m_render_graph.reset();
		m_render_graph = nullptr;
		rebuild();
		m_update = false;
	}
}

void Renderer::onPostTick()
{
	if (!m_render_graph || Window::instance()->isMinimized())
	{
		return;
	}

	m_render_graph->execute(GraphicsContext::instance()->getCurrentCommandBuffer());
	m_render_graph->present(GraphicsContext::instance()->getCurrentCommandBuffer(), GraphicsContext::instance()->getSwapchain().getImages()[GraphicsContext::instance()->getFrameIndex()]);
}

void Renderer::onShutdown()
{
	GraphicsContext::instance()->getQueueSystem().waitAll();
	m_samplers.clear();
	m_buffers.clear();
}

const RenderGraph *Renderer::getRenderGraph() const
{
	return m_render_graph.get();
}

RenderGraph *Renderer::getRenderGraph()
{
	return m_render_graph.get();
}

ResourceCache &Renderer::getResourceCache()
{
	return *m_resource_cache;
}

void Renderer::resetBuilder()
{
	buildRenderGraph = defaultBuilder;
}

void Renderer::rebuild()
{
	GraphicsContext::instance()->getQueueSystem().waitAll();

	m_render_graph.reset();
	m_render_graph = nullptr;

	m_rg_builder.reset();

	buildRenderGraph(m_rg_builder);

	if (m_imgui)
	{
		ImGuiContext::flush();

		m_rg_builder.addRenderPass("ImGuiPass", createScope<pass::ImGuiPass>("imgui_output", m_rg_builder.view(), AttachmentState::Clear_Color)).setOutput("imgui_output");
	}

	m_render_graph = m_rg_builder.build();
	Event_RenderGraph_Rebuild.invoke();
}

bool Renderer::hasImGui() const
{
	return m_imgui;
}

void Renderer::setImGui(bool enable)
{
	if (m_imgui != enable)
	{
		m_imgui = enable;
		rebuild();
		enable ? ImGuiContext::initialize() : ImGuiContext::destroy();
	}
}

const Sampler &Renderer::getSampler(SamplerType type) const
{
	return m_samplers.at(type);
}

const BufferReference Renderer::getBuffer(BufferType type) const
{
	return m_buffers.at(type);
}

const VkExtent2D &Renderer::getRenderTargetExtent() const
{
	return m_render_target_extent;
}

void Renderer::resizeRenderTarget(VkExtent2D extent)
{
	if (m_render_target_extent.height != extent.height || m_render_target_extent.width != extent.width)
	{
		m_render_target_extent = extent;
		m_update               = true;
	}
}

const ImageReference Renderer::getDefaultTexture() const
{
	return m_default_texture;
}

void Renderer::createSamplers()
{
	m_samplers[SamplerType::Compare_Depth]     = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_NEAREST);
	m_samplers[SamplerType::Point_Clamp]       = Sampler(VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_NEAREST);
	m_samplers[SamplerType::Point_Wrap]        = Sampler(VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_NEAREST);
	m_samplers[SamplerType::Bilinear_Clamp]    = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_NEAREST);
	m_samplers[SamplerType::Bilinear_Wrap]     = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_NEAREST);
	m_samplers[SamplerType::Trilinear_Clamp]   = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR);
	m_samplers[SamplerType::Trilinear_Wrap]    = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_LINEAR);
	m_samplers[SamplerType::Anisptropic_Clamp] = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR);
	m_samplers[SamplerType::Anisptropic_Wrap]  = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_LINEAR);
}

void Renderer::updateBuffers()
{
	// Update main camera
	auto *camera_buffer            = reinterpret_cast<CameraBuffer *>(m_buffers[BufferType::MainCamera].map());
	camera_buffer->position        = Main_Camera.position;
	camera_buffer->view_projection = Main_Camera.view_projection;
	m_buffers[BufferType::MainCamera].unmap();

	// Update lights
	std::vector<cmpt::DirectionalLight::Data> directional_lights;
	std::vector<cmpt::SpotLight::Data>              spot_lights;
	std::vector<cmpt::PointLight::Data>       point_lights;

	//// Gather light infos
	const auto group = Scene::instance()->getRegistry().group<>(entt::get<cmpt::Light, cmpt::Tag>);
	group.each([&](const entt::entity &entity, const cmpt::Light &light, const cmpt::Tag &tag) {
		if (!tag.active || !light.impl)
		{
			return;
		}

		switch (light.type)
		{
			case cmpt::LightType::Directional:
				directional_lights.push_back(static_cast<cmpt::DirectionalLight *>(light.impl.get())->data);
				break;
			case cmpt::LightType::Spot:
				spot_lights.push_back(static_cast<cmpt::SpotLight *>(light.impl.get())->data);
				spot_lights.back().position = Entity(entity).getComponent<cmpt::Transform>().translation;
				break;
			case cmpt::LightType::Point:
				point_lights.push_back(static_cast<cmpt::PointLight *>(light.impl.get())->data);
				point_lights.back().position = Entity(entity).getComponent<cmpt::Transform>().translation;
				break;
			default:
				break;
		}
	});

	//// Enlarge buffer
	size_t directional_lights_count = m_buffers[BufferType::DirectionalLight].getSize() / sizeof(cmpt::DirectionalLight::Data);
	if (directional_lights_count < directional_lights.size())
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		m_buffers[BufferType::DirectionalLight] = Buffer(2 * directional_lights_count * sizeof(cmpt::DirectionalLight::Data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		m_update = true;
	}

	size_t spot_lights_count = m_buffers[BufferType::SpotLight].getSize() / sizeof(cmpt::SpotLight::Data);
	if (spot_lights_count < spot_lights.size())
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		m_buffers[BufferType::SpotLight] = Buffer(2 * spot_lights_count * sizeof(cmpt::SpotLight::Data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		m_update = true;
	}

	size_t point_lights_count = m_buffers[BufferType::PointLight].getSize() / sizeof(cmpt::PointLight::Data);
	if (point_lights_count < point_lights.size())
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		m_buffers[BufferType::PointLight] = Buffer(2 * point_lights_count * sizeof(cmpt::PointLight::Data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		m_update = true;
	}

	//// Copy buffer
	if (!directional_lights.empty())
	{
		std::memcpy(m_buffers[BufferType::DirectionalLight].map(), directional_lights.data(), directional_lights.size() * sizeof(cmpt::DirectionalLight::Data));
		m_buffers[BufferType::DirectionalLight].unmap();
	}
	if (!spot_lights.empty())
	{
		std::memcpy(m_buffers[BufferType::SpotLight].map(), spot_lights.data(), spot_lights.size() * sizeof(cmpt::SpotLight::Data));
		m_buffers[BufferType::SpotLight].unmap();
	}
	if (!point_lights.empty())
	{
		std::memcpy(m_buffers[BufferType::PointLight].map(), point_lights.data(), point_lights.size() * sizeof(cmpt::PointLight::Data));
		m_buffers[BufferType::PointLight].unmap();
	}
}

void Renderer::createBuffers()
{
	m_buffers[BufferType::MainCamera]       = Buffer(sizeof(CameraBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::DirectionalLight] = Buffer(2 * sizeof(cmpt::DirectionalLight::Data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::SpotLight]        = Buffer(2 * sizeof(cmpt::SpotLight::Data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::PointLight]       = Buffer(2 * sizeof(cmpt::PointLight::Data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
}
}        // namespace Ilum