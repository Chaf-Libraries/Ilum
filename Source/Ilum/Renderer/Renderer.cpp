#include "Renderer.hpp"
#include "RenderGraph/RenderGraph.hpp"

#include "Renderer/RenderPass/ImGuiPass.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"
#include "Graphics/RenderFrame.hpp"

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

#include "RenderPass/Copy/CopyHizBuffer.hpp"
#include "RenderPass/Copy/CopyLastFrame.hpp"
#include "RenderPass/CopyPass.hpp"

#include "RenderPass/Culling/HizPass.hpp"
#include "RenderPass/Culling/InstanceCullingPass.hpp"
#include "RenderPass/Culling/MeshletCullingPass.hpp"

#include "RenderPass/GeometryView/CurvePass.hpp"
#include "RenderPass/GeometryView/MeshPass.hpp"
#include "RenderPass/GeometryView/SurfacePass.hpp"
#include "RenderPass/GeometryView/WireFrame.hpp"

#include "RenderPass/PostProcess/BloomBlend.hpp"
#include "RenderPass/PostProcess/BloomBlur.hpp"
#include "RenderPass/PostProcess/BloomMask.hpp"
#include "RenderPass/PostProcess/TAA.hpp"
#include "RenderPass/PostProcess/Tonemapping.hpp"

#include "RenderPass/PreProcess/KullaContyAverage.hpp"
#include "RenderPass/PreProcess/KullaContyEnergy.hpp"
#include "RenderPass/Preprocess/EquirectangularToCubemap.hpp"
#include "RenderPass/PreProcess/CubemapSHProjection.hpp"

#include "RenderPass/Shading/Deferred/GeometryPass.hpp"
#include "RenderPass/Shading/Deferred/LightPass.hpp"
#include "RenderPass/Shading/Shadow/CascadeShadowmap.hpp"
#include "RenderPass/Shading/Shadow/OmniShadowmap.hpp"
#include "RenderPass/Shading/Shadow/Shadowmap.hpp"
#include "RenderPass/Shading/SkyboxPass.hpp"

#include "BufferUpdate/CameraUpdate.hpp"
#include "BufferUpdate/CurveUpdate.hpp"
#include "BufferUpdate/GeometryUpdate.hpp"
#include "BufferUpdate/LightUpdate.hpp"
#include "BufferUpdate/MaterialUpdate.hpp"
#include "BufferUpdate/MeshletUpdate.hpp"
#include "BufferUpdate/SurfaceUpdate.hpp"
#include "BufferUpdate/TransformUpdate.hpp"

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
		    .addRenderPass("KullaContyEnergy", std::make_unique<pass::KullaContyEnergy>())
		    .addRenderPass("KullaContyAverage", std::make_unique<pass::KullaContyAverage>())
		    .addRenderPass("EquirectangularToCubemap", std::make_unique<pass::EquirectangularToCubemap>())
		    .addRenderPass("CubemapSHProjection", std::make_unique<pass::CubemapSHProjection>())
		    .addRenderPass("HizPass", std::make_unique<pass::HizPass>())
		    .addRenderPass("InstanceCulling", std::make_unique<pass::InstanceCullingPass>())
		    .addRenderPass("MeshletCulling", std::make_unique<pass::MeshletCullingPass>())
		    .addRenderPass("GeometryPass", std::make_unique<pass::GeometryPass>())
		    .addRenderPass("ShadowmapPass", std::make_unique<pass::ShadowmapPass>())
		    .addRenderPass("CascadeShadowmapPass", std::make_unique<pass::CascadeShadowmapPass>())
		    .addRenderPass("OmniShadowmapPass", std::make_unique<pass::OmniShadowmapPass>())
		    .addRenderPass("LightPass", std::make_unique<pass::LightPass>())
		    .addRenderPass("Skybox", std::make_unique<pass::SkyboxPass>())
		    .addRenderPass("TAAPass", std::make_unique<pass::TAAPass>())
		    .addRenderPass("BloomMask", std::make_unique<pass::BloomMask>("TAAOutput", "PostTex1"))
		    .addRenderPass("BloomBlur1", std::make_unique<pass::BloomBlur>("PostTex1", "PostTex2", false))
		    .addRenderPass("BloomBlur2", std::make_unique<pass::BloomBlur>("PostTex2", "PostTex1", true))
		    .addRenderPass("Blend", std::make_unique<pass::BloomBlend>("PostTex1", "TAAOutput"))
		    .addRenderPass("CopyHizBuffer", std::make_unique<pass::CopyHizBuffer>())
		    .addRenderPass("CopyLastFrame", std::make_unique<pass::CopyLastFrame>("TAAOutput"))
		    .addRenderPass("Tonemapping", std::make_unique<pass::Tonemapping>("TAAOutput"))
		    .addRenderPass("CurvePass", std::make_unique<pass::CurvePass>())
		    .addRenderPass("SurfacePass", std::make_unique<pass::SurfacePass>())
		    .addRenderPass("MeshPass", std::make_unique<pass::MeshPass>())
		    .addRenderPass("WireFramePass", std::make_unique<pass::WireFramePass>())
		    .setView("TAAOutput")
		    .setOutput("TAAOutput");
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
	}
}

void Renderer::onPostTick()
{
	if (!m_render_graph || Window::instance()->isMinimized())
	{
		return;
	}

	auto &cmd_buffer = GraphicsContext::instance()->getFrame().requestCommandBuffer();
	cmd_buffer.begin();
	GraphicsContext::instance()->getProfiler().beginFrame(cmd_buffer);

	m_render_graph->execute();
	m_render_graph->present(cmd_buffer, GraphicsContext::instance()->getSwapchain().getImages()[GraphicsContext::instance()->getFrameIndex()]);

	cmd_buffer.end();
	GraphicsContext::instance()->submitCommandBuffer(cmd_buffer);
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
	return Main_Camera && (Main_Camera.hasComponent<cmpt::PerspectiveCamera>() || Main_Camera.hasComponent<cmpt::OrthographicCamera>());
}

void Renderer::update()
{
	m_update = true;
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
}        // namespace Ilum