#pragma once

#include <type_traits>

namespace Ilum
{
template <typename T>
class Singleton
{
  protected:
	Singleton() = default;

  public:
	static T &GetInstance() noexcept(std::is_nothrow_constructible<T>::value)
	{
		static T instance;
		return instance;
	}

	virtual ~Singleton() = default;

	Singleton(const Singleton &) = delete;
	Singleton(Singleton &&)      = delete;
	Singleton &operator=(const Singleton &) = delete;
	Singleton &operator=(Singleton &&) = delete;
};
}        // namespace Ilum