#pragma once

#include "Precompile.hpp"

#include <spdlog/spdlog.h>

namespace Ilum
{
class  LogSystem final
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

	static LogSystem &GetInstance();

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