#pragma once

#include "Renderable.hpp"

namespace Ilum
{
namespace Cmpt
{
class EXPORT_API MeshRenderer : public Renderable
{
  public:
	MeshRenderer(Node *node);

	virtual void OnImGui() override;

	virtual std::type_index GetType() const override;
};
}        // namespace Cmpt
}        // namespace Ilum