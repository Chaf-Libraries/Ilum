#pragma once

#include <spdlog/sinks/base_sink.h>

#include <deque>
#include <mutex>

namespace Ilum
{
class Sink : public spdlog::sinks::base_sink<std::mutex>
{
  public:
	// Thread unsafe
	const std::deque<std::string> &getLogs() const;

	// Thread safe
	std::vector<std::string> copyLogs() const;

	void clear();

  protected:
	virtual void sink_it_(const spdlog::details::log_msg &msg) override;
	virtual void flush_() override;

  private:
	std::deque<std::string> m_logs;
};
}        // namespace Ilum