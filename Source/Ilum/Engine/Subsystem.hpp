#pragma once

#include <memory>
#include <typeindex>

namespace Ilum
{
enum class TickType
{
	Variable,
	Smoothed
};

class Context;

class ISubsystem
{
  public:
	ISubsystem(Context *context = nullptr);

	virtual ~ISubsystem() = default;

	virtual std::type_index type() const = 0;

	virtual bool onInitialize();

	virtual void onPreTick();

	virtual void onTick(float delta_time);

	virtual void onPostTick();

	virtual void onShutdown();

  protected:
	Context *m_context = nullptr;

	static bool s_enable;
};

template <typename T>
struct TSubsystem : public ISubsystem
{
  public:
	TSubsystem(Context *context = nullptr) :
	    m_context(context)
	{
		s_instance = static_cast<T *>(this);
	}

	static T* instance()
	{
		return s_instance;
	}

	virtual std::type_index type() const override
	{
		return typeid(T);
	}

  protected:
	Context *m_context = nullptr;
	inline static T *s_instance = nullptr;
};

struct Subsystem
{
  public:
	Subsystem(std::unique_ptr<ISubsystem> &&subsystem, TickType tick_group) :
	    ptr(std::move(subsystem)), tick_group(tick_group)
	{
	}

	std::unique_ptr<ISubsystem> ptr;
	TickType                    tick_group;
};
}        // namespace Ilum