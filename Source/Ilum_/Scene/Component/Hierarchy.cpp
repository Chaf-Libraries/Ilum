#include "Hierarchy.hpp"
#include "Transform.hpp"

#include "Scene/Entity.hpp"

namespace Ilum::cmpt
{
void Hierarchy::Tick(Scene &scene, entt::entity entity, RHIDevice *device)
{
	if (m_update)
	{
		Entity(scene, entity).GetComponent<cmpt::Transform>().Update();
		m_update = false;
	}
}

bool Hierarchy::OnImGui(ImGuiContext &context)
{
	ImGui::Text("Parent: %s", m_parent == entt::null ? "false" : "true");
	ImGui::Text("Children: %s", m_first == entt::null ? "false" : "true");
	ImGui::Text("Siblings: %s", m_next == entt::null && m_prev == entt::null ? "false" : "true");
	return m_update;
}

entt::entity Hierarchy::GetParent() const
{
	return m_parent;
}

entt::entity Hierarchy::GetFirst() const
{
	return m_first;
}

entt::entity Hierarchy::GetNext() const
{
	return m_next;
}

entt::entity Hierarchy::GetPrev() const
{
	return m_prev;
}

void Hierarchy::SetParent(entt::entity handle)
{
	m_parent = handle;
	Update();
}

void Hierarchy::SetFirst(entt::entity handle)
{
	m_first = handle;
	Update();
}

void Hierarchy::SetNext(entt::entity handle)
{
	m_next = handle;
	Update();
}

void Hierarchy::SetPrev(entt::entity handle)
{
	m_prev = handle;
	Update();
}

bool Hierarchy::IsRoot() const
{
	return m_parent == entt::null;
}

}        // namespace Ilum::cmpt