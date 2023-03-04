#include "Components/Light/Light.hpp"

namespace Ilum
{
namespace Cmpt
{
Light::Light(const char *name, Node *node) :
    Component(name, node)
{
}

bool Light::CastShadow() const
{
	return false;
}

void Light::SetShadowID(uint32_t &shadow_id)
{
}

void Light::CalculateFrustum(const glm::mat4 &view_projection, std::array<glm::vec4, 6> &frustum)
{
	// Left
	frustum[0].x = view_projection[0].w + view_projection[0].x;
	frustum[0].y = view_projection[1].w + view_projection[1].x;
	frustum[0].z = view_projection[2].w + view_projection[2].x;
	frustum[0].w = view_projection[3].w + view_projection[3].x;

	// Right
	frustum[1].x = view_projection[0].w - view_projection[0].x;
	frustum[1].y = view_projection[1].w - view_projection[1].x;
	frustum[1].z = view_projection[2].w - view_projection[2].x;
	frustum[1].w = view_projection[3].w - view_projection[3].x;

	// Top
	frustum[2].x = view_projection[0].w - view_projection[0].y;
	frustum[2].y = view_projection[1].w - view_projection[1].y;
	frustum[2].z = view_projection[2].w - view_projection[2].y;
	frustum[2].w = view_projection[3].w - view_projection[3].y;

	// Bottom
	frustum[3].x = view_projection[0].w + view_projection[0].y;
	frustum[3].y = view_projection[1].w + view_projection[1].y;
	frustum[3].z = view_projection[2].w + view_projection[2].y;
	frustum[3].w = view_projection[3].w + view_projection[3].y;

	// Near
	frustum[4].x = view_projection[0].w + view_projection[0].z;
	frustum[4].y = view_projection[1].w + view_projection[1].z;
	frustum[4].z = view_projection[2].w + view_projection[2].z;
	frustum[4].w = view_projection[3].w + view_projection[3].z;

	// Far
	frustum[5].x = view_projection[0].w - view_projection[0].z;
	frustum[5].y = view_projection[1].w - view_projection[1].z;
	frustum[5].z = view_projection[2].w - view_projection[2].z;
	frustum[5].w = view_projection[3].w - view_projection[3].z;

	for (auto &plane : frustum)
	{
		float length = glm::length(glm::vec3(plane.x, plane.y, plane.z));
		plane /= length;
	}
}

}        // namespace Cmpt
}        // namespace Ilum