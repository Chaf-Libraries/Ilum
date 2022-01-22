#include "Logger.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>

namespace Ilum::Core
{
std::shared_ptr<spdlog::logger> Logger::s_core_logger = nullptr;
std::vector<spdlog::sink_ptr>   sinks;

void Logger::Initialize()
{
	sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());

	// Create core logger
	s_core_logger = std::make_shared<spdlog::logger>("Ilum", sinks.begin(), sinks.end());
	spdlog::register_logger(s_core_logger);

	spdlog::set_pattern("%^[%T] %v%$");
}

void Logger::Release()
{
	s_core_logger.reset();
	sinks.clear();
	spdlog::shutdown();
}

std::shared_ptr<spdlog::logger> &Logger::GetCoreLogger()
{
	return s_core_logger;
}

void Logger::AddSink(spdlog::sink_ptr &sink)
{
	s_core_logger->sinks().push_back(sink);
	s_core_logger->set_pattern("%^[%T] %v%$");
}
}        // namespace Ilum::Core