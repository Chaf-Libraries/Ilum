#pragma once

#include "Precompile.hpp"

#include "Log.hpp"
#include "Serialization.hpp"

#include <cassert>

#include <rttr/registration.h>

#include <cereal/types/array.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#define SERIALIZER_TYPE_JSON 0
#define SERIALIZER_TYPE_BINARY 1
#define SERIALIZER_TYPE_XML 2
#define SERIALIZER_TYPE SERIALIZER_TYPE_BINARY

#define LOG_HELPER(LOG_LEVEL, ...) \
	Ilum::LogSystem::GetInstance().Log(LOG_LEVEL, "[" + std::string(__FUNCTION__) + "] " + __VA_ARGS__);

// Logging
#define LOG_DEBUG(...) LOG_HELPER(Ilum::LogSystem::LogLevel::Debug, __VA_ARGS__);
#define LOG_INFO(...) LOG_HELPER(Ilum::LogSystem::LogLevel::Info, __VA_ARGS__);
#define LOG_WARN(...) LOG_HELPER(Ilum::LogSystem::LogLevel::Warn, __VA_ARGS__);
#define LOG_ERROR(...) LOG_HELPER(Ilum::LogSystem::LogLevel::Error, __VA_ARGS__);
#define LOG_FATAL(...) LOG_HELPER(Ilum::LogSystem::LogLevel::Fatal, __VA_ARGS__);

#if SERIALIZER_TYPE == SERIALIZER_TYPE_JSON
#	include <cereal/archives/json.hpp>
using InputArchive  = cereal::JSONInputArchive;
using OutputArchive = cereal::JSONOutputArchive;
#elif SERIALIZER_TYPE == SERIALIZER_TYPE_BINARY
#	include <cereal/archives/binary.hpp>
using InputArchive  = cereal::BinaryInputArchive;
using OutputArchive = cereal::BinaryOutputArchive;
#elif SERIALIZER_TYPE == SERIALIZER_TYPE_XML
#	include <cereal/archives/xml.hpp>
using InputArchive  = cereal::XMLInputArchive;
using OutputArchive = cereal::XMLOutputArchive;
#else
#	error Must specify a type of serializer!
#endif

#define SERIALIZE(...)
#define DESERIALIZE(...)

//#define SERIALIZE(FILE, DATA, ...)                \
//	{                                             \
//		std::ofstream os(FILE, std::ios::binary); \
//		OutputArchive archive(os);                \
//		archive(DATA, __VA_ARGS__);               \
//	}
//
//#define DESERIALIZE(FILE, DATA, ...)              \
//	{                                             \
//		std::ifstream is(FILE, std::ios::binary); \
//		InputArchive  archive(is);                \
//		archive(DATA, __VA_ARGS__);               \
//	}

#if defined(__REFLECTION_PARSER__)
#	define META(...) __attribute__((annotate(#    __VA_ARGS__)))
#	define CLASS(class_name, ...) class __attribute__((annotate(#    __VA_ARGS__))) class_name
#	define STRUCT(struct_name, ...) struct __attribute__((annotate(#    __VA_ARGS__))) struct_name
#	define ENUM(enum_name, ...) enum class __attribute__((annotate(#    __VA_ARGS__))) enum_name
#else
#	define META(...)
#	define CLASS(class_name, ...) class class_name
#	define STRUCT(struct_name, ...) struct struct_name
#	define ENUM(enum_name, ...) enum class enum_name
#endif        // __REFLECTION_PARSER__

#ifdef NDEBUG
#	define ASSERT(x)
#else
#	define ASSERT(x) assert(x)
#endif        // NDBUG
