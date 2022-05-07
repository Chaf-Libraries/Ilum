#pragma once

#include <RHI/ImGuiContext.hpp>

#include "Serialize.hpp"

namespace Ilum::cmpt
{
struct Component
{
	virtual bool OnImGui(ImGuiContext &context) = 0;
	virtual void Tick()
	{}
};
}        // namespace Ilum::cmpt