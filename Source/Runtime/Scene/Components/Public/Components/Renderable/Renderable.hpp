#pragma once

#include <SceneGraph/Component.hpp>

namespace Ilum
{
namespace Cmpt
{
class EXPORT_API Renderable : public Component
{
  public:
	Renderable(const char *name, Node *node);

	virtual ~Renderable() = default;

	virtual void OnImGui() override;

	virtual std::type_index GetType() const = 0;

	void AddMaterial(const std::string &material);

	void AddSubmesh(const std::string &submesh);

	void AddAnimation(const std::string &animation);

	const std::vector<std::string> &GetSubmeshes() const;

	const std::vector<std::string> &GetMaterials() const;

	const std::vector<std::string> &GetAnimations() const;

  protected:
	std::vector<std::string> m_submeshes;
	std::vector<std::string> m_materials;
	std::vector<std::string> m_animations;
};
}        // namespace Cmpt
}        // namespace Ilum