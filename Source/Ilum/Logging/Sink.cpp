#include "Sink.hpp"

#include <spdlog/details/log_msg_buffer.h>

#include <iostream>

namespace Ilum
{
const std::deque<Sink::LogMsg> &Sink::getLogs() const
{
	return m_log_msgs;
}

// Thread safe
std::vector<Sink::LogMsg> Sink::copyLogs() const
{
	std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(base_sink<std::mutex>::mutex_));
	std::vector<LogMsg>         res(m_log_msgs.begin(), m_log_msgs.end());
	return res;
}

void Sink::clear()
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
}        // namespace Ilum