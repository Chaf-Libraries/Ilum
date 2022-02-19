#include "SurfaceUpdate.hpp"

#include "Scene/Component/Renderable.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Geometry/Surface/BSplineSurface.hpp"
#include "Geometry/Surface/BezierSurface.hpp"
#include "Geometry/Surface/RationalBSplineSurface.hpp"
#include "Geometry/Surface/RationalBezierSurface.hpp"
#include "Geometry/Surface/Surface.hpp"

#include "Graphics/GraphicsContext.hpp"

#include <tbb/tbb.h>

namespace Ilum::sym
{
void SurfaceUpdate::run()
{
	// Update surface
	auto surface_view = Scene::instance()->getRegistry().view<cmpt::SurfaceRenderer>();
	tbb::parallel_for_each(surface_view.begin(), surface_view.end(), [](entt::entity entity) {
		auto &surface_renderer = Entity(entity).getComponent<cmpt::SurfaceRenderer>();

		scope<geometry::Surface> surface = nullptr;
		// Update vertices
		if (surface_renderer.need_update)
		{
			switch (surface_renderer.type)
			{
				case cmpt::SurfaceType::BezierSurface:
					surface = createScope<geometry::BezierSurface>();
					surface->generateVertices(surface_renderer.vertices, surface_renderer.indices, surface_renderer.control_points, surface_renderer.sample_x, surface_renderer.sample_y);
					break;
				case cmpt::SurfaceType::BSplineSurface:
					surface                                                       = createScope<geometry::BSplineSurface>();
					static_cast<geometry::BSplineSurface *>(surface.get())->order = surface_renderer.order;
					surface->generateVertices(surface_renderer.vertices, surface_renderer.indices, surface_renderer.control_points, surface_renderer.sample_x, surface_renderer.sample_y);
					break;
				case cmpt::SurfaceType::RationalBezierSurface:
					surface                                                                = createScope<geometry::RationalBezierSurface>();
					static_cast<geometry::RationalBezierSurface *>(surface.get())->weights = surface_renderer.weights;
					surface->generateVertices(surface_renderer.vertices, surface_renderer.indices, surface_renderer.control_points, surface_renderer.sample_x, surface_renderer.sample_y);
					break;
				case cmpt::SurfaceType::RationalBSplineSurface:
					surface                                                                 = createScope<geometry::RationalBSplineSurface>();
					static_cast<geometry::RationalBSplineSurface *>(surface.get())->weights = surface_renderer.weights;
					static_cast<geometry::RationalBSplineSurface *>(surface.get())->order   = surface_renderer.order;
					surface->generateVertices(surface_renderer.vertices, surface_renderer.indices, surface_renderer.control_points, surface_renderer.sample_x, surface_renderer.sample_y);
					break;
				default:
					break;
			}
		}

		if (surface_renderer.vertices.size() * sizeof(Vertex) != surface_renderer.vertex_buffer.getSize())
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			surface_renderer.vertex_buffer = Buffer(surface_renderer.vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			surface_renderer.index_buffer  = Buffer(surface_renderer.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

			surface_renderer.need_update = true;
		}

		if (surface_renderer.vertices.size() == 0)
		{
			surface_renderer.need_update = false;
			return;
		}

		// Copy memory
		if (surface_renderer.need_update)
		{
			std::memcpy(surface_renderer.vertex_buffer.map(), surface_renderer.vertices.data(), surface_renderer.vertex_buffer.getSize());
			surface_renderer.vertex_buffer.unmap();

			std::memcpy(surface_renderer.index_buffer.map(), surface_renderer.indices.data(), surface_renderer.index_buffer.getSize());
			surface_renderer.index_buffer.unmap();

			surface_renderer.need_update = false;
		}
	});
}
}        // namespace Ilum::sym