#include "Renderer.hpp"
#include "RenderGraph/RenderGraph.hpp"

#include "Renderer/RenderPass/ImGuiPass.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "File/FileSystem.hpp"

#include "Loader/ImageLoader/Bitmap.hpp"
#include "Loader/ImageLoader/ImageLoader.hpp"

#include "Scene/Component/Camera.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "RenderPass/Compute/HizPass.hpp"
#include "RenderPass/Compute/InstanceCullingPass.hpp"
#include "RenderPass/Compute/MeshletCullingPass.hpp"
#include "RenderPass/CopyPass.hpp"
#include "RenderPass/Deferred/LightPass.hpp"
#include "RenderPass/Deferred/StaticGeometryPass.hpp"
#include "RenderPass/PostProcess/BlendPass.hpp"
#include "RenderPass/PostProcess/BloomPass.hpp"
#include "RenderPass/PostProcess/BlurPass.hpp"
#include "RenderPass/PostProcess/BrightPass.hpp"
#include "RenderPass/PostProcess/TonemappingPass.hpp"

#include "PreProcess/CameraUpdate.hpp"
#include "PreProcess/LightUpdate.hpp"
#include "PreProcess/TransformUpdate.hpp"

#include "Threading/ThreadPool.hpp"

#include <imgui.h>

#include <tbb/tbb.h>

namespace Ilum
{
Renderer::Renderer(Context *context) :
    TSubsystem<Renderer>(context)
{
	GraphicsContext::instance()->Swapchain_Rebuild_Event += [this]() { m_update = true; };

	DeferredRendering = [this](RenderGraphBuilder &builder) {
		builder
		    .addRenderPass("HizPass", std::make_unique<Ilum::pass::HizPass>())
		    .addRenderPass("InstanceCulling", std::make_unique<Ilum::pass::InstanceCullingPass>())
		    .addRenderPass("MeshletCulling", std::make_unique<Ilum::pass::MeshletCullingPass>())
		    .addRenderPass("StaticGeometryPass", std::make_unique<Ilum::pass::StaticGeometryPass>())
		    .addRenderPass("LightPass", std::make_unique<Ilum::pass::LightPass>())
		    .addRenderPass("BrightPass", std::make_unique<Ilum::pass::BrightPass>("lighting"))
		    .addRenderPass("Blur1", std::make_unique<Ilum::pass::BlurPass>("bright", "blur1"))
		    .addRenderPass("Blur2", std::make_unique<Ilum::pass::BlurPass>("blur1", "blur2", true))
		    .addRenderPass("Blend", std::make_unique<Ilum::pass::BlendPass>("blur2", "lighting", "blooming"))
		    .addRenderPass("Tonemapping", std::make_unique<Ilum::pass::TonemappingPass>("blooming"))
		    .addRenderPass("CopyBuffer", std::make_unique<Ilum::pass::CopyPass>())

		    .setView("gbuffer - normal")
		    .setOutput("gbuffer - normal");
	};

	buildRenderGraph = DeferredRendering;

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
	Scene::instance()->addSystem<sym::TransformUpdate>();
	Scene::instance()->addSystem<sym::LightUpdate>();
	Scene::instance()->addSystem<sym::CameraUpdate>();

	m_render_target_extent = GraphicsContext::instance()->getSwapchain().getExtent();
	updateImages();

	DeferredRendering(m_rg_builder);

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
	//Main_Camera_.onUpdate();

	// Check out images update
	if (m_texture_count != m_resource_cache->getImages().size())
	{
		m_update        = true;
		m_texture_count = static_cast<uint32_t>(m_resource_cache->getImages().size());
		m_resource_cache->updateImageReferences();
	}

	if (m_update)
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
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

void Renderer::rebuild()
{
	GraphicsContext::instance()->getQueueSystem().waitAll();

	updateImages();

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

bool Renderer::hasMainCamera()
{
	return Main_Camera && (Main_Camera.hasComponent < cmpt::PerspectiveCamera>()||Main_Camera.hasComponent<cmpt::OrthographicCamera>());
}

void Renderer::update()
{
	m_update = true;
}

void Renderer::updateGeometry()
{
	auto &vertex_buffer = m_buffers[BufferType::Vertex];
	auto &index_buffer  = m_buffers[BufferType::Index];

	if (m_resource_cache->getVerticesCount() == 0)
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		m_buffers[BufferType::Vertex] = Buffer(0, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
		m_buffers[BufferType::Index]  = Buffer(0, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	}
	else
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		vertex_buffer = Buffer(m_resource_cache->getVerticesCount() * sizeof(Ilum::Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
		index_buffer  = Buffer(m_resource_cache->getIndicesCount() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

		Buffer staging_vertex_buffer(vertex_buffer.getSize(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		Buffer staging_index_buffer(index_buffer.getSize(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		auto * vertex_data = staging_vertex_buffer.map();
		auto * index_data  = staging_index_buffer.map();

		// CPU -> Staging
		for (auto &[name, index] : m_resource_cache->getModels())
		{
			auto &model = m_resource_cache->loadModel(name);

			std::memcpy(vertex_data + model.get().vertices_offset * sizeof(Ilum::Vertex), model.get().mesh.vertices.data(), sizeof(Ilum::Vertex) * model.get().vertices_count);
			std::memcpy(index_data + model.get().indices_offset * sizeof(uint32_t), model.get().mesh.indices.data(), sizeof(uint32_t) * model.get().indices_count);
		}

		staging_vertex_buffer.unmap();
		staging_index_buffer.unmap();

		// Staging -> GPU
		CommandBuffer command_buffer(QueueUsage::Transfer);
		command_buffer.begin();
		command_buffer.copyBuffer(BufferInfo{staging_vertex_buffer}, BufferInfo{vertex_buffer}, vertex_buffer.getSize());
		command_buffer.copyBuffer(BufferInfo{staging_index_buffer}, BufferInfo{index_buffer}, index_buffer.getSize());
		command_buffer.end();
		command_buffer.submitIdle();
	}
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


	updateInstanceBuffer();
}

void Renderer::updateInstanceBuffer()
{
	GraphicsContext::instance()->getProfiler().beginSample("Instance Update");
	m_update = m_update || Render_Queue.update();
	GraphicsContext::instance()->getProfiler().endSample("Instance Update");
}

void Renderer::updateImages()
{
	Renderer::instance()->Last_Frame.hiz_buffer   = createScope<Image>(Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height,
                                                                     VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VMA_MEMORY_USAGE_GPU_ONLY, true);
	Renderer::instance()->Last_Frame.depth_buffer = createScope<Image>(Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height,
	                                                                   VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	// Layout transition
	{
		CommandBuffer cmd_buffer;
		cmd_buffer.begin();
		cmd_buffer.transferLayout(*Renderer::instance()->Last_Frame.hiz_buffer, VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.transferLayout(*Renderer::instance()->Last_Frame.depth_buffer, VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.end();
		cmd_buffer.submitIdle();
	}
}

void Renderer::createBuffers()
{
	//m_buffers[BufferType::MainCamera]      = Buffer(1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::Material]        = Buffer(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::Transform]       = Buffer(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::IndirectCommand] = Buffer(1, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::BoundingBox]     = Buffer(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::Meshlet]         = Buffer(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::Vertex]          = Buffer(0, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	m_buffers[BufferType::Index]           = Buffer(0, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}
}        // namespace Ilum