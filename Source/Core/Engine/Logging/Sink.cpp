#include "Sink.h"

#include <spdlog/details/log_msg_buffer.h>

#include <iostream>

namespace Ilum
{
const std::deque<std::string> &Sink::getLogs() const
{
	return m_logs;
}

// Thread safe
std::vector<std::string> Sink::copyLogs() const
{
	std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(base_sink<std::mutex>::mutex_));
	std::vector<std::string>    res(m_logs.begin(), m_logs.end());
	return res;
}

void Sink::clear()
{
	std::lock_guard<std::mutex> lock(base_sink<std::mutex>::mutex_);
	m_logs.clear();
}

void Sink::sink_it_(const spdlog::details::log_msg &msg)
{
	spdlog::details::log_msg_buffer buffer(msg);
	spdlog::memory_buf_t            formatted;
	base_sink<std::mutex>::formatter_->format(buffer, formatted);
	m_logs.push_back(fmt::to_string(formatted));
	std::cout << fmt::to_string(formatted);
}

void Sink::flush_()
{
	std::cout << std::flush;
}
}        // namespace Ilum