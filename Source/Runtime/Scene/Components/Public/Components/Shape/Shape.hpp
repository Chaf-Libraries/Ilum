#pragma once

#include <SceneGraph/Component.hpp>

namespace Ilum
{
namespace Cmpt
{
class EXPORT_API Shape : public Component
{
  public:
	Shape(const char *name, Node *node);

	virtual void OnImGui() = 0;

	virtual std::type_index GetType() const = 0;


};
}        // namespace Cmpt
}        // namespace Ilum