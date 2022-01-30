#include "RHICommand.hpp"

namespace Ilum::RHI
{
RHICommand::RHICommand(const CmdUsage &usage, const std::thread::id &thread_id):
    m_usage(usage), m_thread_id(thread_id)
{
}

 std::shared_ptr<RHICommand> RHICommand::Create(const CmdUsage &usage, const std::thread::id &thread_id)
{
	 return CreateFunc(usage, thread_id);
 }
}        // namespace Ilum::RHI