#pragma once

namespace Ilum
{
class NonCopyable
{
  protected:
	NonCopyable()  = default;
	~NonCopyable() = default;

  public:
	NonCopyable(NonCopyable &) = delete;
	NonCopyable &operator=(NonCopyable &) = delete;
	NonCopyable(NonCopyable &&)           = default;
	NonCopyable &operator=(NonCopyable &&) = default;
};
}        // namespace Ilum