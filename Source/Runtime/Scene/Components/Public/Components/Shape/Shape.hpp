#pragma once

#include <SceneGraph/Component.hpp>

namespace Ilum
{
namespace Cmpt
{
class Shape : public Component
{
  public:
	Shape(Node *node);

	virtual void OnImGui() = 0;

	virtual std::type_index GetType() const = 0;


};
}        // namespace Cmpt
}        // namespace Ilum