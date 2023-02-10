#pragma once

#include "Renderable.hpp"

namespace Ilum
{
namespace Cmpt
{
class SkinnedMeshRenderer : public Renderable
{
  public:
	SkinnedMeshRenderer(Node *node);

	virtual ~SkinnedMeshRenderer() = default;

	virtual bool OnImGui() override;

	virtual std::type_index GetType() const override;
};
}        // namespace Cmpt
}        // namespace Ilum