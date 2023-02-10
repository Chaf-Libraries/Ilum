#pragma once

#include <rttr/registration.h>

namespace Ilum
{
STRUCT(Component, Enable)
{
	META(Editor("Disable"))
	bool update = false;

	RTTR_ENABLE();
};
}        // namespace Ilum