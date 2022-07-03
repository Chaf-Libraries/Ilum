#pragma once

#include "Singleton.hpp"

#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <stdexcept>

namespace Ilum
{
class Sink : public spdlog::sinks::base_sink<std::mutex>
{
  public:
	struct LogMsg
	{
		spdlog::level::level_enum level;
		std::string               msg;
	};

  public:
	// Thread unsafe
	const std::deque<LogMsg> &GetLogs() const;

	// Thread safe
	std::vector<LogMsg> CopyLogs() const;

	void Clear();

  protected:
	virtual void sink_it_(const spdlog::details::log_msg &msg) override;
	virtual void flush_() override;

  private:
	std::deque<LogMsg> m_log_msgs;
};

class LogSystem final : public Singleton<LogSystem>
{
  public:
	enum class LogLevel : uint8_t
	{
		Debug,
		Info,
		Warn,
		Error,
		Fatal
	};

  public:
	LogSystem();
	~LogSystem();

	template <typename... Args>
	void Log(LogLevel level, Args &&...args)
	{
		switch (level)
		{
			case Ilum::LogSystem::LogLevel::Debug:
				m_logger->debug(std::forward<Args>(args)...);
				break;
			case Ilum::LogSystem::LogLevel::Info:
				m_logger->info(std::forward<Args>(args)...);
				break;
			case Ilum::LogSystem::LogLevel::Warn:
				m_logger->warn(std::forward<Args>(args)...);
				break;
			case Ilum::LogSystem::LogLevel::Error:
				m_logger->error(std::forward<Args>(args)...);
				break;
			case Ilum::LogSystem::LogLevel::Fatal:
				m_logger->critical(std::forward<Args>(args)...);
				throw std::runtime_error(fmt::format(std::forward<Args>(args)...));
				break;
			default:
				break;
		}
	}

  private:
	std::shared_ptr<spdlog::logger> m_logger;
};
}        // namespace Ilum