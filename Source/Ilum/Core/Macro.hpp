#pragma once

#include "Log.hpp"
#include "Serialization.hpp"

#include <cassert>

#include <rttr/registration.h>

#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/string.hpp>

#define SERIALIZER_TYPE_JSON 0
#define SERIALIZER_TYPE_BINARY 1
#define SERIALIZER_TYPE_XML 2
#define SERIALIZER_TYPE SERIALIZER_TYPE_JSON

#define LOG_HELPER(LOG_LEVEL, ...) \
	Ilum::LogSystem::GetInstance().Log(LOG_LEVEL, "[" + std::string(__FUNCTION__) + "] " + __VA_ARGS__);

// Logging
#define LOG_DEBUG(...) LOG_HELPER(Ilum::LogSystem::LogLevel::Debug, __VA_ARGS__);
#define LOG_INFO(...) LOG_HELPER(Ilum::LogSystem::LogLevel::Info, __VA_ARGS__);
#define LOG_WARN(...) LOG_HELPER(Ilum::LogSystem::LogLevel::Warn, __VA_ARGS__);
#define LOG_ERROR(...) LOG_HELPER(Ilum::LogSystem::LogLevel::Error, __VA_ARGS__);
#define LOG_FATAL(...) LOG_HELPER(Ilum::LogSystem::LogLevel::Fatal, __VA_ARGS__);

#define RTTR_REGISTERATION_BEGIN(TYPE)             \
	namespace NAMESPACE_ RTTR_REGISTERATION_##TYPE \
	{                                              \
		RTTR_REGISTRATION                          \
		{
#define RTTR_REGISTERATION_END() \
	}                            \
	}

#if SERIALIZER_TYPE == SERIALIZER_TYPE_JSON
#include <cereal/archives/json.hpp>
using InputSerializer=Ilum::TSerializer<cereal::JSONInputArchive>;
using OutputSerializer=Ilum::TSerializer<cereal::JSONOutputArchive>;
#elif SERIALIZER_TYPE == SERIALIZER_TYPE_BINARY
#include <cereal/archives/binary.hpp>
using InputSerializer=Ilum::TSerializer<cereal::BinaryInputArchive>;
using OutputSerializer=Ilum::TSerializer<cereal::BinaryOutputArchive>;
#elif SERIALIZER_TYPE == SERIALIZER_TYPE_XML
#include <cereal/archives/xml.hpp>
using InputSerializer=Ilum::TSerializer<cereal::XMLInputArchive>;
using OutputSerializer=Ilum::TSerializer<cereal::XMLOutputArchive>;
#else
#	error Must specify a type of serializer!
#endif

#ifdef NDEBUG
#	define ASSERT(x)
#else
#	define ASSERT(x) assert(x)
#endif        // NDBUG
