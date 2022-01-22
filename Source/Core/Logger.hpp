#pragma once

#pragma warning(push, 0)
#include <spdlog/spdlog.h>
#pragma warning(pop)

namespace Ilum::Core
{
class Logger
{
  public:
	static void Initialize();

	static void Release();

	static std::shared_ptr<spdlog::logger> &GetCoreLogger();

	static void AddSink(spdlog::sink_ptr &sink);

  private:
	static std::shared_ptr<spdlog::logger> s_core_logger;
};
}        // namespace Ilum::Core

#define LOG_TRACE(...) SPDLOG_LOGGER_CALL(::Ilum::Core::Logger::GetCoreLogger(), spdlog::level::level_enum::trace, __VA_ARGS__)
#define LOG_INFO(...) SPDLOG_LOGGER_CALL(::Ilum::Core::Logger::GetCoreLogger(), spdlog::level::level_enum::info, __VA_ARGS__)
#define LOG_WARN(...) SPDLOG_LOGGER_CALL(::Ilum::Core::Logger::GetCoreLogger(), spdlog::level::level_enum::warn, __VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_LOGGER_CALL(::Ilum::Core::Logger::GetCoreLogger(), spdlog::level::level_enum::err, __VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_LOGGER_CALL(::Ilum::Core::Logger::GetCoreLogger(), spdlog::level::level_enum::critical, __VA_ARGS__)