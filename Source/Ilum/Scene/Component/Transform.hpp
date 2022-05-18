#pragma once

#include "Component.hpp"

#include <glm/glm.hpp>

namespace Ilum::cmpt
{
class Transform : public Component
{
  public:
	Transform() = default;

	bool OnImGui(ImGuiContext &context);

	virtual void Tick(Scene &scene, entt::entity entity, RHIDevice *device) override;

	const glm::vec3 &GetTranslation() const;
	const glm::vec3 &GetRotation() const;
	const glm::vec3 &GetScale() const;
	const glm::mat4 &GetLocalTransform() const;
	const glm::mat4 &GetWorldTransform() const;

	void SetTranslation(const glm::vec3 &translation);
	void SetRotation(const glm::vec3 &rotation);
	void SetScale(const glm::vec3 &scale);

	template <class Archive>
	void serialize(Archive &ar)
	{
		glm::serialize(ar, m_translation);
		glm::serialize(ar, m_rotation);
		glm::serialize(ar, m_scale);
	}

  private:
	glm::vec3 m_translation = {0.f, 0.f, 0.f};
	glm::vec3 m_rotation    = {0.f, 0.f, 0.f};
	glm::vec3 m_scale       = {1.f, 1.f, 1.f};

	glm::mat4 m_local_transform = glm::mat4(1.f);
	glm::mat4 m_world_transform = glm::mat4(1.f);
};
}        // namespace Ilum::cmpt