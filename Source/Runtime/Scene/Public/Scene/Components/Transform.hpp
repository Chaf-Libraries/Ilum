#pragma once

#include <Scene/Component.hpp>

#include <glm/glm.hpp>

namespace Ilum
{
class Node;

namespace Cmpt
{
class Transform : public Component
{
  public:
	Transform(Node *node);

	~Transform() = default;

	virtual void OnImGui() override;

	virtual std::type_index GetType() const override;

	virtual void Save(OutputArchive &archive) const override;

	virtual void Load(InputArchive &archive) override;

	const glm::vec3 &GetTranslation() const;

	const glm::vec3 &GetRotation() const;

	const glm::vec3 &GetScale() const;

	const glm::mat4 GetLocalTransform() const;

	const glm::mat4 GetWorldTransform();

	void SetTranslation(const glm::vec3 &translation);

	void SetRotation(const glm::vec3 &rotation);

	void SetScale(const glm::vec3 &scale);

	void SetDirty();

  private:
	void Update();

  private:
	glm::vec3 m_translation = {0.f, 0.f, 0.f};
	glm::vec3 m_rotation    = {0.f, 0.f, 0.f};
	glm::vec3 m_scale       = {1.f, 1.f, 1.f};

	glm::mat4 m_world_transform = glm::mat4(1.f);

	bool m_dirty = false;
};
}        // namespace Cmpt
}        // namespace Ilum