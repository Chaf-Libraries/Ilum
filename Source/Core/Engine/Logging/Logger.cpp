#include "Logger.h"
#include "Sink.h"

#include <fstream>

#include <spdlog/sinks/base_sink.h>

namespace Ilum
{
Logger &Logger::getInstance()
{
	static Logger instance;
	return instance;
}

Logger::~Logger()
{
	clear();
	for (auto &logger : m_loggers)
	{
		logger.second.reset();
	}
	m_loggers.clear();
}

const std::deque<std::string> Logger::getLogs(const std::string &name)
{
	if (!valid(name))
	{
		return {};
	}

	auto &logger = m_loggers.at(name);
	auto  sink   = std::dynamic_pointer_cast<Sink>(logger->sinks().front());
	return sink->getLogs();
}

const std::vector<std::string> Logger::copyLogs(const std::string &name)
{
	if (!valid(name))
	{
		return {};
	}

	auto &logger = m_loggers.at(name);
	auto  sink   = std::dynamic_pointer_cast<Sink>(logger->sinks().front());
	return sink->copyLogs();
}

void Logger::clear(const std::string &name)
{
	if (!valid(name))
	{
		return;
	}
	auto &logger = m_loggers.at(name);
	auto  sink   = std::dynamic_pointer_cast<Sink>(logger->sinks().front());
	sink->clear();
}

void Logger::clear()
{
	for (auto &logger : m_loggers)
	{
		clear(logger.first);
	}
}

void Logger::save(const std::string &name)
{
	if (!valid(name))
	{
		return;
	}

	auto &      logger = m_loggers.at(name);
	auto        sink   = std::dynamic_pointer_cast<Sink>(logger->sinks().front());
	const auto &logs   = sink->getLogs();

	std::ofstream oft(name + ".log");
	if (oft.fail())
	{
		log(name, spdlog::level::err, "Log saving failed!");
		return;
	}
	for (const auto &log : logs)
	{
		oft << log.c_str();
	}
	oft.close();
}

void Logger::save()
{
	for (auto &logger : m_loggers)
	{
		save(logger.first);
	}
}

bool Logger::valid(const std::string &name)
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