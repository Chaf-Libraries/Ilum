#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class Semaphore
{
  public:
	enum class State
	{
		Idle,
		Submitted,
		Signaled
	};

  public:
	Semaphore(bool timeline = false);

	~Semaphore();

	bool wait(const uint64_t value, const uint64_t timeout = std::numeric_limits<uint64_t>::max()) const;

	bool signal(const uint64_t value) const;

	bool isTimeline() const;

	// Only when is timeline
	uint64_t count() const;

	State getState() const;

	const VkSemaphore &getSemaphore() const;

	operator const VkSemaphore &() const;

  private:
	VkSemaphore m_handle   = VK_NULL_HANDLE;
	bool        m_timeline = false;
	State       m_state    = State::Idle;
};
}        // namespace Ilum