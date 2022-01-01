#include "CurveUpdate.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"

#include "Scene/Component/Renderable.hpp"
#include "Scene/Scene.hpp"

#include "Geometry/Curve/BezierCurve.hpp"
#include "Geometry/Curve/BezierSpline.hpp"
#include "Geometry/Curve/BSpline.hpp"

#include <tbb/tbb.h>

namespace Ilum::sym
{
void CurveUpdate::run()
{
	// Update curve
	auto curve_view = Scene::instance()->getRegistry().view<cmpt::CurveRenderer>();
	tbb::parallel_for_each(curve_view.begin(), curve_view.end(), [](entt::entity entity) {
		auto &curve_renderer = Entity(entity).getComponent<cmpt::CurveRenderer>();

		scope<geometry::Curve> curve;
		// Update vertices
		if (curve_renderer.need_update)
		{
			switch (curve_renderer.type)
			{
				case cmpt::CurveType::None:
					curve_renderer.vertices.clear();
					break;
				case cmpt::CurveType::BezierCurve:
					curve                   = createScope<geometry::BezierCurve>();
					curve_renderer.vertices = std::move(curve->generateVertices(curve_renderer.control_points, curve_renderer.sample));
					break;
				case cmpt::CurveType::BezierSpline:
					curve                   = createScope<geometry::BezierSpline>();
					curve_renderer.vertices = std::move(curve->generateVertices(curve_renderer.control_points, curve_renderer.sample));
					break;
				case cmpt::CurveType::BSpline:
					curve                   = createScope<geometry::BSpline>();
					curve_renderer.vertices = std::move(curve->generateVertices(curve_renderer.control_points, curve_renderer.sample));
				default:
					break;
			}
		}

		if (curve_renderer.vertices.size() * sizeof(glm::vec3) != curve_renderer.vertex_buffer.getSize())
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			curve_renderer.vertex_buffer = Buffer(curve_renderer.vertices.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

			curve_renderer.need_update = true;
		}

		if (curve_renderer.vertices.size() == 0)
		{
			curve_renderer.need_update = false;
			return;
		}

		// Copy memory
		if (curve_renderer.need_update)
		{
			std::memcpy(curve_renderer.vertex_buffer.map(), curve_renderer.vertices.data(), curve_renderer.vertex_buffer.getSize());
			curve_renderer.vertex_buffer.unmap();

			curve_renderer.need_update = false;
		}
	});
}
}        // namespace Ilum::sym