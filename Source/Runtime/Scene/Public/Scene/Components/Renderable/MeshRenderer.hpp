#pragma once

#include "Renderable.hpp"

namespace Ilum
{
namespace Cmpt
{
class MeshRenderer : public Renderable
{
  public:
	MeshRenderer(Node *node);

	virtual ~MeshRenderer() = default;

	virtual void OnImGui() override;

	virtual std::type_index GetType() const override;
};
}        // namespace Cmpt
}        // namespace Ilum