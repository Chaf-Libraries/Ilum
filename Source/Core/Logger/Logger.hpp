#pragma once

#include "Sink.hpp"

#define SPDLOG_WCHAR_FILENAMES

#include <spdlog/spdlog.h>

#include <deque>

namespace Ilum::Core
{
class Logger
{
  public:
	static Logger &GetInstance();

	~Logger();

	// Thread unsafe
	const std::deque<Sink::LogMsg> GetLogs(const std::string &name);

	// Thread safe
	const std::vector<Sink::LogMsg> CopyLogs(const std::string &name);

	void Clear(const std::string &name);

	void Clear();

	void Save(const std::string &name);

	void Save();

	template <typename FormatString, typename... Args>
	void Log(const std::string &name, spdlog::level::level_enum lvl, const FormatString &fmt, const Args... args)
	{
		if (!Valid(name))
		{
			return;
		}
		auto &logger = m_loggers.at(name);
		logger->log(lvl, fmt, args...);
	}

	template <typename FormatString, typename... Args>
	void Debug(const std::string &name, bool x, const FormatString &fmt, const Args... args)
	{
		if (!Valid(name))
		{
			return;
		}
		if (!x)
		{
			auto &logger = m_loggers.at(name);
			logger->log(spdlog::level::debug, fmt, args...);
#ifdef _DEBUG
			__debugbreak();
#endif        // _DEBUG
		}
	}

  private:
	bool Valid(const std::string &name);

	std::unordered_map<std::string, std::shared_ptr<spdlog::logger>> m_loggers;
};
}        // namespace Ilum

#define LOG_INFO(...) Ilum::Core::Logger::GetInstance().Log("engine", spdlog::level::info, __VA_ARGS__);
#define LOG_WARN(...) Ilum::Core::Logger::GetInstance().Log("engine", spdlog::level::warn, __VA_ARGS__);
#define LOG_ERROR(...) Ilum::Core::Logger::GetInstance().Log("engine", spdlog::level::err, __VA_ARGS__);
#define LOG_TRACE(...) Ilum::Core::Logger::GetInstance().Log("engine", spdlog::level::trace, __VA_ARGS__);
#define LOG_CRITICAL(...) Ilum::Core::Logger::GetInstance().Log("engine", spdlog::level::critical, __VA_ARGS__);
#define LOG_DEBUG(x, ...) Ilum::Core::Logger::GetInstance().debug("engine", x, __VA_ARGS__);

#define VK_INFO(...) Ilum::Core::Logger::GetInstance().Log("vulkan", spdlog::level::info, __VA_ARGS__);
#define VK_WARN(...) Ilum::Core::Logger::GetInstance().Log("vulkan", spdlog::level::warn, __VA_ARGS__);
#define VK_ERROR(...) Ilum::Core::Logger::GetInstance().Log("vulkan", spdlog::level::err, __VA_ARGS__);
#define VK_TRACE(...) Ilum::Core::Logger::GetInstance().Log("vulkan", spdlog::level::trace, __VA_ARGS__);
#define VK_CRITICAL(...) Ilum::Core::Logger::GetInstance().Log("vulkan", spdlog::level::critical, __VA_ARGS__);
#define VK_DEBUG(x, ...) Ilum::Core::Logger::GetInstance().debug("vulkan", x, __VA_ARGS__);

#ifdef _DEBUG
#	define ASSERT(expression) assert(expression)
#else
#	define ASSERT(expression)      \
		if (!(##expression))        \
		{                           \
			LOG_ERROR(#expression); \
			__debugbreak();         \
		}
#endif        // _DEBUG