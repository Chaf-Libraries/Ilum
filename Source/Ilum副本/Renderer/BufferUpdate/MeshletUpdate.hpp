#pragma once

#include "Utils/PCH.hpp"

#include "Scene/System.hpp"

namespace Ilum::sym
{
class MeshletUpdate : public System
{
  public:
	MeshletUpdate() = default;

	~MeshletUpdate() = default;

	virtual void run() override;
};
}        // namespace Ilum::sym