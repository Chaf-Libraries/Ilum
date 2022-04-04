#pragma once

#include <entt.hpp>

#include "Scene/Entity.hpp"

#include <cereal/cereal.hpp>

namespace Ilum::cmpt
{
struct Hierarchy
{
	entt::entity parent = entt::null;
	entt::entity first  = entt::null;
	entt::entity next   = entt::null;
	entt::entity prev   = entt::null;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(parent, first, next, prev);
	}
};
}        // namespace Ilum::cmpt