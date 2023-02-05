#pragma once

#include <Scene/Component.hpp>

namespace Ilum
{
namespace Cmpt
{
class Renderable : public Component
{
  public:
	Renderable(const char *name, Node *node);

	virtual ~Renderable() = default;

	virtual bool OnImGui() override;

	virtual std::type_index GetType() const = 0;

	virtual void Save(OutputArchive &archive) const override;

	virtual void Load(InputArchive &archive) override;

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