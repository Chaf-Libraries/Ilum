#pragma once

#include "Component.hpp"

namespace Ilum
{
struct HierarchyComponent : public Component
{
	entt::entity parent = entt::null;
	entt::entity first  = entt::null;
	entt::entity next   = entt::null;
	entt::entity prev   = entt::null;
};
}        // namespace Ilum