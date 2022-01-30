#pragma once

#include <functional>
#include <thread>

namespace Ilum::RHI
{
enum class CmdUsage
{
	Graphics,
	Compute,
	Transfer
};

class RHICommand
{
  public:
	RHICommand(const CmdUsage &usage, const std::thread::id &thread_id);

	virtual ~RHICommand(){};

	static std::shared_ptr<RHICommand> Create(const CmdUsage &usage = CmdUsage::Graphics, const std::thread::id &thread_id = std::this_thread::get_id());

	virtual void Begin() = 0;

	virtual void End() = 0;

	virtual void Reset() = 0;

  private:
	static std::function<std::shared_ptr<RHICommand>(const CmdUsage &, const std::thread::id &)> CreateFunc;

  protected:
	CmdUsage        m_usage;
	std::thread::id m_thread_id;
};
}        // namespace Ilum::RHI