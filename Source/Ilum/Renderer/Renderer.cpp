#include "Renderer.hpp"
#include "RenderGraph/RenderGraph.hpp"

#include "Renderer/RenderPass/ImGuiPass.hpp"

#include "Device/Swapchain.hpp"
#include <Graphics/Device/Device.hpp>

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"

#include "ImGui/ImGuiContext.hpp"

#include <Core/FileSystem.hpp>

#include "Loader/ImageLoader/Bitmap.hpp"
#include "Loader/ImageLoader/ImageLoader.hpp"

#include "Scene/Component/Camera.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "RenderPass/CopyPass.hpp"
#include "RenderPass/Culling/HizPass.hpp"
#include "RenderPass/Culling/InstanceCullingPass.hpp"
#include "RenderPass/Culling/MeshletCullingPass.hpp"
#include "RenderPass/Deferred/DynamicGeometryPass.hpp"
#include "RenderPass/Deferred/EnvLightPass.hpp"
#include "RenderPass/Deferred/LightPass.hpp"
#include "RenderPass/Deferred/StaticGeometryPass.hpp"
#include "RenderPass/GeometryView/CurvePass.hpp"
#include "RenderPass/GeometryView/SurfacePass.hpp"
#include "RenderPass/PostProcess/BlendPass.hpp"
#include "RenderPass/PostProcess/BloomPass.hpp"
#include "RenderPass/PostProcess/BlurPass.hpp"
#include "RenderPass/PostProcess/BrightPass.hpp"
#include "RenderPass/PostProcess/TAA.hpp"
#include "RenderPass/PostProcess/TonemappingPass.hpp"
#include "RenderPass/PreProcess/KullaContyAverage.hpp"
#include "RenderPass/PreProcess/KullaContyEnergy.hpp"
#include "RenderPass/Preprocess/EquirectangularToCubemap.hpp"

#include "BufferUpdate/CameraUpdate.hpp"
#include "BufferUpdate/CurveUpdate.hpp"
#include "BufferUpdate/GeometryUpdate.hpp"
#include "BufferUpdate/LightUpdate.hpp"
#include "BufferUpdate/MaterialUpdate.hpp"
#include "BufferUpdate/MeshletUpdate.hpp"
#include "BufferUpdate/SurfaceUpdate.hpp"
#include "BufferUpdate/TransformUpdate.hpp"

#include <Core/JobSystem/JobSystem.hpp>

#include <Graphics/Device/Window.hpp>
#include <Graphics/RenderContext.hpp>

#include <imgui.h>

#include <tbb/tbb.h>

namespace Ilum
{
Renderer::Renderer(Context *context) :
    TSubsystem<Renderer>(context),
    m_default_texture(Graphics::RenderContext::GetDevice())
{
	GraphicsContext::instance()->Swapchain_Rebuild_Event += [this]() { m_update = true; };

	DeferredRendering = [this](RenderGraphBuilder &builder) {
		builder
		    .addRenderPass("KullaContyEnergy", std::make_unique<pass::KullaContyEnergy>())
		    .addRenderPass("KullaContyAverage", std::make_unique<pass::KullaContyAverage>())
		    .addRenderPass("EquirectangularToCubemap", std::make_unique<pass::EquirectangularToCubemap>())
		    .addRenderPass("HizPass", std::make_unique<pass::HizPass>())
		    .addRenderPass("InstanceCulling", std::make_unique<pass::InstanceCullingPass>())
		    .addRenderPass("MeshletCulling", std::make_unique<pass::MeshletCullingPass>())
		    .addRenderPass("StaticGeometryPass", std::make_unique<pass::StaticGeometryPass>())
		    .addRenderPass("DynamicGeometryPass", std::make_unique<pass::DynamicGeometryPass>())
		    .addRenderPass("CurvePass", std::make_unique<pass::CurvePass>())
		    .addRenderPass("SurfacePass", std::make_unique<pass::SurfacePass>())
		    .addRenderPass("LightPass", std::make_unique<pass::LightPass>())
		    .addRenderPass("EnvLight", std::make_unique<pass::EnvLightPass>())
		    .addRenderPass("TAAPass", std::make_unique<pass::TAAPass>())
		    .addRenderPass("CopyBuffer", std::make_unique<pass::CopyPass>())

		    //.addRenderPass("BrightPass", std::make_unique<pass::BrightPass>("lighting"))
		    //.addRenderPass("Blur1", std::make_unique<pass::BlurPass>("bright", "blur1"))
		    //.addRenderPass("Blur2", std::make_unique<pass::BlurPass>("blur1", "blur2", true))
		    //.addRenderPass("Blend", std::make_unique<pass::BlendPass>("blur2", "lighting", "output"))

		    .addRenderPass("Tonemapping", std::make_unique<pass::TonemappingPass>("taa_result"))

		    .setView("gbuffer - normal")
		    .setOutput("gbuffer - normal");
	};

	buildRenderGraph = DeferredRendering;

	m_resource_cache = createScope<ResourceCache>();
	createSamplers();
	ImageLoader::loadImage(m_default_texture, Bitmap{{0, 0, 0, 255}, VK_FORMAT_R8G8B8A8_UNORM, 1, 1}, false);
}

Renderer::~Renderer()
{
}

bool Renderer::onInitialize()
{
	Scene::instance()->addSystem<sym::GeometryUpdate>();
	Scene::instance()->addSystem<sym::CurveUpdate>();
	Scene::instance()->addSystem<sym::SurfaceUpdate>();
	Scene::instance()->addSystem<sym::TransformUpdate>();
	Scene::instance()->addSystem<sym::LightUpdate>();
	Scene::instance()->addSystem<sym::CameraUpdate>();
	Scene::instance()->addSystem<sym::MeshletUpdate>();
	Scene::instance()->addSystem<sym::MaterialUpdate>();

	m_render_target_extent = GraphicsContext::instance()->getSwapchain().getExtent();
	updateImages();

	DeferredRendering(m_rg_builder);

	rebuild();

	return true;
}

void Renderer::onPreTick()
{
}

void Renderer::onTick(float delta_time)
{
	// Flush resource cache
	m_resource_cache->flush();

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

		EnvLight.update = true;
	}
}

void Renderer::onPostTick()
{
	if (!m_render_graph || Graphics::RenderContext::GetWindow().IsMinimized())
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
	Event_RenderGraph_Rebuild.Invoke();
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

const Graphics::Sampler &Renderer::getSampler(SamplerType type) const
{
	return m_samplers.at(type);
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

const Graphics::ImageReference Renderer::getDefaultTexture() const
{
	return m_default_texture;
}

bool Renderer::hasMainCamera()
{
	return Main_Camera && (Main_Camera.hasComponent<cmpt::PerspectiveCamera>() || Main_Camera.hasComponent<cmpt::OrthographicCamera>());
}

void Renderer::update()
{
	m_update = true;
}

void Renderer::createSamplers()
{
	m_samplers.emplace(SamplerType::Compare_Depth, Graphics::Sampler(Graphics::RenderContext::GetDevice(), VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_NEAREST));
	m_samplers.emplace(SamplerType::Point_Clamp, Graphics::Sampler(Graphics::RenderContext::GetDevice(), VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_NEAREST));
	m_samplers.emplace(SamplerType::Point_Wrap, Graphics::Sampler(Graphics::RenderContext::GetDevice(), VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_NEAREST));
	m_samplers.emplace(SamplerType::Bilinear_Clamp, Graphics::Sampler(Graphics::RenderContext::GetDevice(), VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_NEAREST));
	m_samplers.emplace(SamplerType::Bilinear_Wrap, Graphics::Sampler(Graphics::RenderContext::GetDevice(), VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_NEAREST));
	m_samplers.emplace(SamplerType::Trilinear_Clamp, Graphics::Sampler(Graphics::RenderContext::GetDevice(), VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR));
	m_samplers.emplace(SamplerType::Trilinear_Wrap, Graphics::Sampler(Graphics::RenderContext::GetDevice(), VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_LINEAR));
	m_samplers.emplace(SamplerType::Anisptropic_Clamp, Graphics::Sampler(Graphics::RenderContext::GetDevice(), VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR));
	m_samplers.emplace(SamplerType::Anisptropic_Wrap, Graphics::Sampler(Graphics::RenderContext::GetDevice(), VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_LINEAR));
}

void Renderer::updateImages()
{
	GraphicsContext::instance()->getQueueSystem().waitAll();

	Renderer::instance()->Last_Frame.hiz_buffer   = createScope<Graphics::Image>(Graphics::RenderContext::GetDevice(), Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height,
                                                                               VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VMA_MEMORY_USAGE_GPU_ONLY, true);
	Renderer::instance()->Last_Frame.depth_buffer = createScope<Graphics::Image>(Graphics::RenderContext::GetDevice(), Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height,
	                                                                             VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	Renderer::instance()->Last_Frame.last_result  = createScope<Graphics::Image>(Graphics::RenderContext::GetDevice(), Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height,
                                                                                VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	// Layout transition
	{
		CommandBuffer cmd_buffer;
		cmd_buffer.begin();
		cmd_buffer.transferLayout(*Renderer::instance()->Last_Frame.hiz_buffer, VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.transferLayout(*Renderer::instance()->Last_Frame.depth_buffer, VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.transferLayout(*Renderer::instance()->Last_Frame.last_result, VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.end();
		cmd_buffer.submitIdle();
	}
}
}        // namespace Ilum