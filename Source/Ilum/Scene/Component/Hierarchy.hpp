#pragma once

#include <entt.hpp>

namespace Ilum::cmpt
{
struct Hierarchy
{
	entt::entity parent;
	entt::entity first;
	entt::entity next;
	entt::entity prev;
};
}