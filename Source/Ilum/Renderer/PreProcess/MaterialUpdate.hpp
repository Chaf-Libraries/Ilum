#pragma once

#include "Utils/PCH.hpp"

#include "Scene/System.hpp"

namespace Ilum::sym
{
class MaterialUpdate : public System
{
  public:
	MaterialUpdate() = default;

	~MaterialUpdate() = default;

	virtual void run() override;
};
}        // namespace Ilum::sym