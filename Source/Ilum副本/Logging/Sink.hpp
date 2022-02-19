#pragma once

#include <spdlog/sinks/base_sink.h>

#include <deque>
#include <mutex>
#include <string>

namespace Ilum
{
class Sink : public spdlog::sinks::base_sink<std::mutex>
{
  public:
	struct LogMsg
	{
		spdlog::level::level_enum level;
		std::string msg;
	};

  public:
	// Thread unsafe
	const std::deque<LogMsg> &getLogs() const;

	// Thread safe
	std::vector<LogMsg> copyLogs() const;

	void clear();

  protected:
	virtual void sink_it_(const spdlog::details::log_msg &msg) override;
	virtual void flush_() override;

  private:
	std::deque<LogMsg> m_log_msgs;
};
}        // namespace Ilum