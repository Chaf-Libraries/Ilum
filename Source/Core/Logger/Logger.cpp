#include "Logger.hpp"

#include <fstream>

#include <spdlog/sinks/base_sink.h>

namespace Ilum::Core
{
Logger &Logger::GetInstance()
{
	static Logger instance;
	return instance;
}

Logger::~Logger()
{
	Clear();
	for (auto &logger : m_loggers)
	{
		logger.second.reset();
	}
	m_loggers.clear();
}

const std::deque<Sink::LogMsg> Logger::GetLogs(const std::string &name)
{
	if (!Valid(name))
	{
		return {};
	}

	auto &logger = m_loggers.at(name);
	auto  sink   = std::dynamic_pointer_cast<Sink>(logger->sinks().front());
	return sink->GetLogs();
}

const std::vector<Sink::LogMsg> Logger::CopyLogs(const std::string &name)
{
	if (!Valid(name))
	{
		return {};
	}

	auto &logger = m_loggers.at(name);
	auto  sink   = std::dynamic_pointer_cast<Sink>(logger->sinks().front());
	return sink->CopyLogs();
}

void Logger::Clear(const std::string &name)
{
	if (!Valid(name))
	{
		return;
	}
	auto &logger = m_loggers.at(name);
	auto  sink   = std::dynamic_pointer_cast<Sink>(logger->sinks().front());
	sink->Clear();
}

void Logger::Clear()
{
	for (auto &logger : m_loggers)
	{
		Clear(logger.first);
	}
}

void Logger::Save(const std::string &name)
{
	if (!Valid(name))
	{
		return;
	}

	auto &      logger = m_loggers.at(name);
	auto        sink   = std::dynamic_pointer_cast<Sink>(logger->sinks().front());
	const auto &logs   = sink->GetLogs();

	std::ofstream oft(name + ".log");
	if (oft.fail())
	{
		Log(name, spdlog::level::err, "Log saving failed!");
		return;
	}
	for (const auto &log : logs)
	{
		oft << log.msg.c_str();
	}
	oft.close();
}

void Logger::Save()
{
	for (auto &logger : m_loggers)
	{
		Save(logger.first);
	}
}

bool Logger::Valid(const std::string &name)
{
	try
	{
		if (m_loggers.find(name) == m_loggers.end())
		{
			m_loggers.insert({name, spdlog::synchronous_factory::create<Sink>(name)});
		}
		return true;
	}
	catch (const std::exception &)
	{
		return false;
	}
}
}        // namespace Ilum
