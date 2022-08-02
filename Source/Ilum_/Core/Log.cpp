#include "Log.hpp"

#include <spdlog/async.h>
#include <spdlog/details/log_msg_buffer.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace Ilum
{
const std::deque<Sink::LogMsg> &Sink::GetLogs() const
{
	return m_log_msgs;
}

// Thread safe
std::vector<Sink::LogMsg> Sink::CopyLogs() const
{
	std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(base_sink<std::mutex>::mutex_));
	std::vector<LogMsg>         res(m_log_msgs.begin(), m_log_msgs.end());
	return res;
}

void Sink::Clear()
{
	std::lock_guard<std::mutex> lock(base_sink<std::mutex>::mutex_);
	m_log_msgs.clear();
}

void Sink::sink_it_(const spdlog::details::log_msg &msg)
{
	spdlog::details::log_msg_buffer buffer(msg);
	spdlog::memory_buf_t            formatted;
	base_sink<std::mutex>::formatter_->format(buffer, formatted);
	m_log_msgs.push_back({msg.level, fmt::to_string(formatted)});
	std::cout << fmt::to_string(formatted);
}

void Sink::flush_()
{
	std::cout << std::flush;
}

LogSystem::LogSystem()
{
	m_logger = spdlog::synchronous_factory::create<Sink>("Ilum");

	m_logger->set_level(spdlog::level::trace);
}

LogSystem::~LogSystem()
{
	m_logger->flush();
	spdlog::drop_all();
}

}        // namespace Ilum