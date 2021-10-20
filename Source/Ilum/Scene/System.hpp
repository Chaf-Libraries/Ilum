#pragma once

#include <future>

namespace Ilum
{
class System
{
  public:
	System() = default;

	~System() = default;

	virtual void run() = 0;
};
}        // namespace Ilum