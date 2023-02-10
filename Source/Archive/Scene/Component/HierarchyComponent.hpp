#pragma once

#include "Component.hpp"

#include <entt.hpp>

namespace Ilum
{
STRUCT(HierarchyComponent, Enable) :
    public Component
{
	uint32_t parent = entt::null;
	uint32_t first  = entt::null;
	uint32_t next   = entt::null;
	uint32_t prev   = entt::null;
};
}        // namespace Ilum