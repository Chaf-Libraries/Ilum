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

	virtual void OnImGui() override =0;

	virtual std::type_index GetType() const override = 0;

	void AddMaterial(size_t uuid);

	void AddSubmesh(size_t uuid);

  protected:
	std::vector<size_t> m_submeshes;
	std::vector<size_t> m_materials;
};
}        // namespace Cmpt
}        // namespace Ilum