#pragma once

#include "Component.hpp"

#include <entt.hpp>

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::cmpt
{
class Hierarchy : public Component
{
  public:
	Hierarchy() = default;

	virtual void Tick(Scene &scene, entt::entity entity, RHIDevice *device) override;

	virtual bool OnImGui(ImGuiContext &context) override;

	entt::entity GetParent() const;
	entt::entity GetFirst() const;
	entt::entity GetNext() const;
	entt::entity GetPrev() const;

	void SetParent(entt::entity handle);
	void SetFirst(entt::entity handle);
	void SetNext(entt::entity handle);
	void SetPrev(entt::entity handle);

	bool IsRoot() const;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(m_parent, m_first, m_next, m_prev);
	}

  private:
	entt::entity m_parent = entt::null;
	entt::entity m_first  = entt::null;
	entt::entity m_next   = entt::null;
	entt::entity m_prev   = entt::null;
};
}        // namespace Ilum::cmpt