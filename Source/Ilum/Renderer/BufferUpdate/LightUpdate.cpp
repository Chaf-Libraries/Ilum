#include "LightUpdate.hpp"

#include "Scene/Component/Light.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"

#include "Renderer/Renderer.hpp"

#include <tbb/tbb.h>

namespace Ilum::sym
{
void LightUpdate::run()
{
	GraphicsContext::instance()->getProfiler().beginSample("Light Update");

	// Collect light infos
	auto directional_lights = Scene::instance()->getRegistry().group<cmpt::DirectionalLight>(entt::get<cmpt::Tag>);
	auto point_lights       = Scene::instance()->getRegistry().group<cmpt::PointLight>(entt::get<cmpt::Tag, cmpt::Transform>);
	auto spot_lights        = Scene::instance()->getRegistry().group<cmpt::SpotLight>(entt::get<cmpt::Tag, cmpt::Transform>);

	// Enlarge buffer
	if (Renderer::instance()->Render_Buffer.Directional_Light_Buffer.GetSize() / sizeof(cmpt::SpotLight) < directional_lights.size())
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		Renderer::instance()->Render_Buffer.Directional_Light_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), directional_lights.size() * sizeof(cmpt::DirectionalLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		Renderer::instance()->update();
	}
	if (Renderer::instance()->Render_Buffer.Spot_Light_Buffer.GetSize() / sizeof(cmpt::SpotLight) < spot_lights.size())
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		Renderer::instance()->Render_Buffer.Spot_Light_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), spot_lights.size() * sizeof(cmpt::SpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		Renderer::instance()->update();
	}
	if (Renderer::instance()->Render_Buffer.Point_Light_Buffer.GetSize() / sizeof(cmpt::PointLight) < point_lights.size())
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		Renderer::instance()->Render_Buffer.Point_Light_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), point_lights.size() * sizeof(cmpt::PointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		Renderer::instance()->update();
	}

	// Copy Graphics::Buffer
	Renderer::instance()->Render_Stats.light_count.directional_light_count = 0;
	directional_lights.each([](entt::entity entity, cmpt::DirectionalLight &light, cmpt::Tag &tag) {
		std::memcpy(reinterpret_cast<cmpt::DirectionalLight *>(Renderer::instance()->Render_Buffer.Directional_Light_Buffer.Map()) + Renderer::instance()->Render_Stats.light_count.directional_light_count++,
		            &light, sizeof(cmpt::DirectionalLight));
	});

	Renderer::instance()->Render_Stats.light_count.point_light_count = 0;
	point_lights.each([](entt::entity entity, cmpt::PointLight &light, cmpt::Tag &tag, cmpt::Transform &transform) {
		light.position = transform.translation;
		std::memcpy(reinterpret_cast<cmpt::PointLight *>(Renderer::instance()->Render_Buffer.Point_Light_Buffer.Map()) + Renderer::instance()->Render_Stats.light_count.point_light_count++,
		            &light, sizeof(cmpt::PointLight));
	});

	Renderer::instance()->Render_Stats.light_count.spot_light_count = 0;
	spot_lights.each([](entt::entity entity, cmpt::SpotLight &light, cmpt::Tag &tag, cmpt::Transform &transform) {
		std::memcpy(reinterpret_cast<cmpt::SpotLight *>(Renderer::instance()->Render_Buffer.Spot_Light_Buffer.Map()) + Renderer::instance()->Render_Stats.light_count.spot_light_count++,
		            &light, sizeof(cmpt::SpotLight));
	});

	Renderer::instance()->Render_Buffer.Directional_Light_Buffer.Unmap();
	Renderer::instance()->Render_Buffer.Spot_Light_Buffer.Unmap();
	Renderer::instance()->Render_Buffer.Point_Light_Buffer.Unmap();

	GraphicsContext::instance()->getProfiler().endSample("Light Update");
}
}        // namespace Ilum::sym