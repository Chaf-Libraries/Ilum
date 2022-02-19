#pragma once

#include <entt.hpp>

#include "Scene/Entity.hpp"

namespace Ilum::cmpt
{
struct Hierarchy
{
	entt::entity parent = entt::null;
	entt::entity first  = entt::null;
	entt::entity next   = entt::null;
	entt::entity prev   = entt::null;
};
}        // namespace Ilum::cmpt