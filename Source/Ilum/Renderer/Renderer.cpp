#include "Renderer.hpp"
#include "RenderGraph/RenderGraph.hpp"

#include "Renderer/RenderPass/ImGuiPass.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Model/Vertex.hpp"
#include "Graphics/Profiler.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "File/FileSystem.hpp"

#include "Loader/ImageLoader/Bitmap.hpp"
#include "Loader/ImageLoader/ImageLoader.hpp"

#include "Scene/Component/DirectionalLight.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/MeshRenderer.hpp"
#include "Scene/Component/PointLight.hpp"
#include "Scene/Component/SpotLight.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Threading/ThreadPool.hpp"

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
		m_buffers[BufferType::Vertex] = Buffer(0, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
		m_buffers[BufferType::Index]  = Buffer(0, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	}
	else
	{
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

			std::memcpy(vertex_data + model.get().vertices_offset * sizeof(Ilum::Vertex), model.get().vertices.data(), sizeof(Ilum::Vertex) * model.get().vertices_count);
			std::memcpy(index_data + model.get().indices_offset * sizeof(uint32_t), model.get().indices.data(), sizeof(uint32_t) * model.get().indices_count);
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
	updateCameraBuffer();

	updateLightBuffer();

	updateInstanceBuffer();
}

void Renderer::updateLightBuffer()
{
	GraphicsContext::instance()->getProfiler().beginSample("Light Update");
	{
		std::vector<cmpt::DirectionalLight::Data> directional_lights;
		std::vector<cmpt::SpotLight::Data>        spot_lights;
		std::vector<cmpt::PointLight::Data>       point_lights;

		// Gather light infos
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
	GraphicsContext::instance()->getProfiler().endSample("Light Update");
}

void Renderer::updateCameraBuffer()
{
	GraphicsContext::instance()->getProfiler().beginSample("Camera Update");
	// Update main camera
	if (Main_Camera.update)
	{
		Main_Camera.onUpdate();

		struct
		{
			glm::mat4 view_projection;
			glm::vec4 frustum[6];
			alignas(16) glm::vec3 position;
		} camera_buffer;

		if (m_buffers[BufferType::MainCamera].getSize() != sizeof(camera_buffer))
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			m_buffers[BufferType::MainCamera] = Buffer(sizeof(camera_buffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		}
		camera_buffer.position        = Main_Camera.position;
		camera_buffer.view_projection = Main_Camera.view_projection;
		for (uint32_t i = 0; i < 6; i++)
		{
			const auto &plane        = Main_Camera.frustum.planes[i];
			camera_buffer.frustum[i] = glm::vec4(plane.normal, plane.constant);
		}
		std::memcpy(m_buffers[BufferType::MainCamera].map(), &camera_buffer, sizeof(camera_buffer));
		m_buffers[BufferType::MainCamera].unmap();
	}

	GraphicsContext::instance()->getProfiler().endSample("Camera Update");
}

void Renderer::updateInstanceBuffer()
{
	GraphicsContext::instance()->getProfiler().beginSample("Instance Update");

	struct MaterialData
	{
		glm::vec4 base_color      = {};
		glm::vec3 emissive_color  = {0.f, 0.f, 0.f};
		float     metallic_factor = 0.f;

		float    roughness_factor   = 0.f;
		float    emissive_intensity = 0.f;
		uint32_t albedo_map         = 0;
		uint32_t normal_map         = 0;

		uint32_t metallic_map  = 0;
		uint32_t roughness_map = 0;
		uint32_t emissive_map  = 0;
		uint32_t ao_map        = 0;

		alignas(16) float displacement_height = 0.f;
		uint32_t displacement_map             = 0;
		uint32_t id                           = 0;
	};

	struct Bound
	{
		glm::vec3 center = {};
		float     radius = 0.f;

		glm::vec3 cone_apex             = {};
		float     cone_cutoff           = 0.f;
		alignas(16) glm::vec3 cone_axis = {};
	};

	struct TransformData
	{
		glm::mat4 world_transform;
		glm::mat4 pre_transform;
	};

	std::atomic<uint32_t> instance_count = 0;

	auto group = Scene::instance()->getRegistry().group<cmpt::MeshRenderer, cmpt::Tag, cmpt::Transform>();
	std::for_each(std::execution::par_unseq, group.begin(), group.end(), [&instance_count, this](auto entity) {
		auto &mesh_renderer = Entity(entity).getComponent<cmpt::MeshRenderer>();
		if (!m_resource_cache->hasModel(mesh_renderer.model))
		{
			return;
		}

		auto &model = m_resource_cache->loadModel(mesh_renderer.model);

		std::for_each(std::execution::par_unseq, mesh_renderer.materials.begin(), mesh_renderer.materials.end(), [&instance_count](scope<IMaterial> &material) {
			if (material->type() == typeid(material::DisneyPBR))
			{
				instance_count.fetch_add(1, std::memory_order_relaxed);
			}
		});
	});

	if (instance_count != Instance_Count || ResourceCache::update || cmpt::MeshRenderer::update)
	{
		cmpt::MeshRenderer::update = false;

		// Collect data
		std::vector<VkDrawIndexedIndirectCommand> indirect_commands;
		std::vector<MaterialData>                 material_data;
		std::vector<TransformData>                transform_data;
		std::vector<Bound>                        bound_data;
		std::vector<uint32_t>                     meshlet_data;        //	Which submesh the meshlet belong

		uint32_t   instance_idx = 0;
		uint32_t   meshlet_idx  = 0;
		const auto group        = Scene::instance()->getRegistry().group<>(entt::get<cmpt::MeshRenderer, cmpt::Tag, cmpt::Transform>);
		group.each([&](const entt::entity &entity, const cmpt::MeshRenderer &mesh_renderer, const cmpt::Tag &tag, const cmpt::Transform &transform) {
			if (!m_resource_cache->hasModel(mesh_renderer.model))
			{
				return;
			}

			auto &model = m_resource_cache->loadModel(mesh_renderer.model);
			for (uint32_t i = 0; i < mesh_renderer.materials.size(); i++)
			{
				auto &submesh      = model.get().submeshes[i];
				auto &material_ptr = mesh_renderer.materials[i];
				if (material_ptr->type() == typeid(material::DisneyPBR))
				{
					auto *material = static_cast<material::DisneyPBR *>(material_ptr.get());

					for (uint32_t j = submesh.meshlet_offset; j < submesh.meshlet_offset + submesh.meshlet_count; j++)
					{
						const auto &meshlet = model.get().meshlets[j];

						meshlet_data.push_back(instance_idx);

						VkDrawIndexedIndirectCommand indirect_cmd;
						indirect_cmd.firstInstance = meshlet_idx++;
						indirect_cmd.firstIndex    = model.get().indices_offset + meshlet.indices_offset;
						indirect_cmd.vertexOffset  = model.get().vertices_offset + meshlet.vertices_offset;
						indirect_cmd.instanceCount = 1;
						indirect_cmd.indexCount    = meshlet.indices_count;
						indirect_commands.push_back(indirect_cmd);

						Bound bound;
						bound.center      = glm::vec3(meshlet.bounds.center[0], meshlet.bounds.center[1], meshlet.bounds.center[2]);
						bound.radius      = meshlet.bounds.radius;
						bound.cone_apex   = glm::vec3(meshlet.bounds.cone_apex[0], meshlet.bounds.cone_apex[1], meshlet.bounds.cone_apex[2]);
						bound.cone_axis   = glm::vec3(meshlet.bounds.cone_axis[0], meshlet.bounds.cone_axis[1], meshlet.bounds.cone_axis[2]);
						bound.cone_cutoff = meshlet.bounds.cone_cutoff;

						bound_data.push_back(bound);
					}

					TransformData trans;
					trans.world_transform = transform.world_transform;
					trans.pre_transform   = submesh.pre_transform;
					transform_data.push_back(trans);

					MaterialData data;
					data.base_color          = material->base_color;
					data.metallic_factor     = material->metallic_factor;
					data.roughness_factor    = material->roughness_factor;
					data.emissive_color      = material->emissive_color;
					data.emissive_intensity  = material->emissive_intensity;
					data.albedo_map          = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->albedo_map));
					data.normal_map          = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->normal_map));
					data.metallic_map        = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->metallic_map));
					data.roughness_map       = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->roughness_map));
					data.emissive_map        = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->emissive_map));
					data.ao_map              = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->ao_map));
					data.displacement_height = material->displacement_height;
					data.displacement_map    = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->displacement_map));
					data.id                  = static_cast<uint32_t>(entity);
					material_data.push_back(data);

					instance_idx++;
				}
			}
		});

		Instance_Count = instance_idx;
		Meshlet_Count  = meshlet_idx;

		// Enlarge buffer
		if (m_buffers[BufferType::IndirectCommand].getSize() == 0)
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			m_buffers[BufferType::Material]        = Buffer(sizeof(MaterialData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			m_buffers[BufferType::Transform]       = Buffer(sizeof(TransformData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			m_buffers[BufferType::Meshlet]         = Buffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			m_buffers[BufferType::BoundingBox]     = Buffer(sizeof(Bound), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			m_buffers[BufferType::IndirectCommand] = Buffer(sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		}

		if (m_buffers[BufferType::IndirectCommand].getSize() < indirect_commands.size() * sizeof(VkDrawIndexedIndirectCommand))
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			m_buffers[BufferType::IndirectCommand] = Buffer(indirect_commands.size() * sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			m_buffers[BufferType::Material]        = Buffer(material_data.size() * sizeof(MaterialData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			m_buffers[BufferType::Transform]       = Buffer(transform_data.size() * sizeof(TransformData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			m_buffers[BufferType::BoundingBox]     = Buffer(bound_data.size() * sizeof(Bound), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			m_buffers[BufferType::Meshlet]         = Buffer(meshlet_data.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

			m_update = true;
		}

		// Copy buffer
		if (!material_data.empty())
		{
			std::memcpy(m_buffers[BufferType::Material].map(), material_data.data(), material_data.size() * sizeof(MaterialData));
			std::memcpy(m_buffers[BufferType::IndirectCommand].map(), indirect_commands.data(), indirect_commands.size() * sizeof(VkDrawIndexedIndirectCommand));
			std::memcpy(m_buffers[BufferType::Transform].map(), transform_data.data(), transform_data.size() * sizeof(TransformData));
			std::memcpy(m_buffers[BufferType::BoundingBox].map(), bound_data.data(), bound_data.size() * sizeof(Bound));
			std::memcpy(m_buffers[BufferType::Meshlet].map(), meshlet_data.data(), meshlet_data.size() * sizeof(uint32_t));
			m_buffers[BufferType::Material].unmap();
			m_buffers[BufferType::IndirectCommand].unmap();
			m_buffers[BufferType::Transform].unmap();
			m_buffers[BufferType::BoundingBox].unmap();
			m_buffers[BufferType::Meshlet].unmap();
		}
	}
	else if (cmpt::Transform::update)
	{
		std::vector<TransformData> transform_data;

		const auto group = Scene::instance()->getRegistry().group<>(entt::get<cmpt::MeshRenderer, cmpt::Tag, cmpt::Transform>);
		group.each([&](const entt::entity &entity, const cmpt::MeshRenderer &mesh_renderer, const cmpt::Tag &tag, const cmpt::Transform &transform) {
			if (!m_resource_cache->hasModel(mesh_renderer.model))
			{
				return;
			}

			auto &model = m_resource_cache->loadModel(mesh_renderer.model);
			for (uint32_t i = 0; i < mesh_renderer.materials.size(); i++)
			{
				auto &submesh      = model.get().submeshes[i];
				auto &material_ptr = mesh_renderer.materials[i];
				if (material_ptr->type() == typeid(material::DisneyPBR))
				{
					TransformData trans;
					trans.world_transform = transform.world_transform;
					trans.pre_transform   = submesh.pre_transform;
					transform_data.push_back(trans);
				}
			}
		});

		// Copy buffer
		std::memcpy(m_buffers[BufferType::Transform].map(), transform_data.data(), transform_data.size() * sizeof(TransformData));
		m_buffers[BufferType::Transform].unmap();
	}

	GraphicsContext::instance()->getProfiler().endSample("Instance Update");
}

void Renderer::createBuffers()
{
	m_buffers[BufferType::MainCamera]       = Buffer(1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::DirectionalLight] = Buffer(2 * sizeof(cmpt::DirectionalLight::Data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::SpotLight]        = Buffer(2 * sizeof(cmpt::SpotLight::Data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::PointLight]       = Buffer(2 * sizeof(cmpt::PointLight::Data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::Material]         = Buffer(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::Transform]        = Buffer(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::IndirectCommand]  = Buffer(1, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::BoundingBox]      = Buffer(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::Meshlet]          = Buffer(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_buffers[BufferType::Vertex]           = Buffer(0, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	m_buffers[BufferType::Index]            = Buffer(0, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}
}        // namespace Ilum