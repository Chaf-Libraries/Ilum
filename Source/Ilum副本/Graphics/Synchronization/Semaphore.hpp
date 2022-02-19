#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class Semaphore
{
  public:
	Semaphore(bool timeline = false);

	~Semaphore();

	Semaphore(const Semaphore &) = delete;

	Semaphore &operator=(const Semaphore &) = delete;

	Semaphore(Semaphore &&other) noexcept;

	Semaphore &operator=(Semaphore &&other) noexcept;

	bool wait(const uint64_t value, const uint64_t timeout = std::numeric_limits<uint64_t>::max()) const;

	bool signal(const uint64_t value) const;

	bool isTimeline() const;

	// Only when is timeline
	uint64_t count() const;

	const VkSemaphore &getSemaphore() const;

	operator const VkSemaphore &() const;

  private:
	VkSemaphore m_handle   = VK_NULL_HANDLE;
	bool        m_timeline = false;
};
}        // namespace Ilum