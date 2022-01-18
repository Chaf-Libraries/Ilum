#pragma once

#include <atomic>

namespace Ilum::Core
{
class SpinLock
{
  public:
	SpinLock();

	~SpinLock() = default;

	void Lock();

	void Unlock();

  private:
	std::atomic_flag m_flag;
};
}        // namespace Ilum::Core