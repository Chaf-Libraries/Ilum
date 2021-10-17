#pragma once

#include <entt.hpp>

namespace Ilum::Cmpt
{
struct Hierarchy
{
	entt::entity parent;
	entt::entity first;
	entt::entity next;
	entt::entity prev;
};
}