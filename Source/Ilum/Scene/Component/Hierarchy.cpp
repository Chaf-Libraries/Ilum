#include "Hierarchy.hpp"

#include "Scene/Entity.hpp"

namespace Ilum::cmpt
{
inline void UpdateTransformRecursive(Scene &scene, entt::entity entity, RHIDevice *device)
{
	if (entity == entt::null)
	{
		return;
	}

	auto &transform = Entity(scene, entity).GetComponent<cmpt::Transform>();
	auto &hierarchy = Entity(scene, entity).GetComponent<cmpt::Hierarchy>();

	transform.Tick(scene, entity, device);

	auto child = hierarchy.GetFirst();

	while (child != entt::null)
	{
		UpdateTransformRecursive(scene, child, device);
		child = Entity(scene, child).GetComponent<cmpt::Hierarchy>().GetNext();
	}
}

void Hierarchy::Tick(Scene &scene, entt::entity entity, RHIDevice *device)
{
	if (m_update)
	{
		UpdateTransformRecursive(scene, entity, device);
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
	m_update = true;
}

void Hierarchy::SetFirst(entt::entity handle)
{
	m_first = handle;
	m_update = true;
}

void Hierarchy::SetNext(entt::entity handle)
{
	m_next = handle;
	m_update = true;
}

void Hierarchy::SetPrev(entt::entity handle)
{
	m_prev = handle;
	m_update = true;
}

bool Hierarchy::IsRoot() const
{
	return m_parent == entt::null;
}

}        // namespace Ilum::cmpt