#include "Renderer.hpp"
#include "RenderGraph/RenderGraph.hpp"

#include "Renderer/RenderPass/DebugPass.hpp"
#include "Renderer/RenderPass/ImGuiPass.hpp"

#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"
#include "Device/LogicalDevice.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"
#include "Loader/ImageLoader/Bitmap.hpp"

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

	// Check out images update
	if (m_texture_count != m_resource_cache->getImages().size())
	{
		m_texture_count = static_cast<uint32_t>(m_resource_cache->getImages().size());
		for (auto& node : m_render_graph->getNodes())
		{
			// Update bindless texture
			node.descriptors.setOption(ResolveOption::Once);
		}
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
	m_samplers.clear();
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
	m_render_graph.reset();
	m_render_graph = nullptr;

	m_rg_builder.reset();

	buildRenderGraph(m_rg_builder);

	if (m_imgui)
	{
		ImGuiContext::flush();

		if (m_debug && !m_rg_builder.empty())
		{
			m_rg_builder.addRenderPass("DebugPass", createScope<pass::DebugPass>());
		}

		m_rg_builder.addRenderPass("ImGuiPass", createScope<pass::ImGuiPass>("imgui_output", m_rg_builder.view(), AttachmentState::Clear_Color)).setOutput("imgui_output");
	}

	m_render_graph = m_rg_builder.build();
	Event_RenderGraph_Rebuild.invoke();
}

bool Renderer::isDebug() const
{
	return m_debug;
}

void Renderer::setDebug(bool enable)
{
	if (m_debug != enable)
	{
		m_debug = enable;
		rebuild();
	}
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
	// Update Main Camera
	auto *camera_buffer = reinterpret_cast<CameraBuffer *>(m_buffers[BufferType::MainCamera].map());
	camera_buffer->position = Main_Camera.position;
	camera_buffer->view_projection = Main_Camera.camera.view_projection;
	m_buffers[BufferType::MainCamera].unmap();

}

void Renderer::createBuffers()
{
	m_buffers[BufferType::MainCamera] = Buffer(sizeof(CameraBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
}
}        // namespace Ilum