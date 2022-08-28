#pragma once

#include "Log.hpp"
#include "Serialization.hpp"

#include <cassert>

#include <rttr/registration.h>

#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#include <fstream>

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

#if SERIALIZER_TYPE == SERIALIZER_TYPE_JSON
#	include <cereal/archives/json.hpp>
using InputSerializer  = Ilum::TSerializer<cereal::JSONInputArchive>;
using OutputSerializer = Ilum::TSerializer<cereal::JSONOutputArchive>;
using InputArchive     = cereal::JSONInputArchive;
using OutputArchive    = cereal::JSONOutputArchive;
#elif SERIALIZER_TYPE == SERIALIZER_TYPE_BINARY
#	include <cereal/archives/binary.hpp>
using InputSerializer  = Ilum::TSerializer<cereal::BinaryInputArchive>;
using OutputSerializer = Ilum::TSerializer<cereal::BinaryOutputArchive>;
#elif SERIALIZER_TYPE == SERIALIZER_TYPE_XML
#	include <cereal/archives/xml.hpp>
using InputSerializer  = Ilum::TSerializer<cereal::XMLInputArchive>;
using OutputSerializer = Ilum::TSerializer<cereal::XMLOutputArchive>;
#else
#	error Must specify a type of serializer!
#endif

#define SERIALIZER_REGISTER_TYPE(TYPE)                                        \
	TSerializer<OutputArchive>::GetInstance().RegisterType<decltype(TYPE)>(); \
	TSerializer<InputArchive>::GetInstance().RegisterType<decltype(TYPE)>();

#define SERIALIZE(FILE, DATA, ...)                     \
	{                                             \
		std::ofstream os(FILE, std::ios::binary); \
		OutputArchive archive(os);                \
		archive(DATA, __VA_ARGS__);                            \
	}

#define DESERIALIZE(FILE, DATA, ...)                   \
	{                                             \
		std::ifstream is(FILE, std::ios::binary); \
		InputArchive  archive(is);                \
		archive(DATA, __VA_ARGS__);                            \
	}

#define META(KEY, VALUE) rttr::metadata(KEY, VALUE)

#define REFLECTION_BEGIN(TYPE, ...)                                      \
	template <class Archive>                                             \
	void serialize(Archive &ar, TYPE &t)                                 \
	{                                                                    \
		rttr::variant var = t;                                           \
		TSerializer<Archive>::GetInstance().Serialize(ar, var);          \
		t = var.convert<TYPE>();                                         \
	}                                                                    \
	namespace RTTR_REGISTRATION_NAMESPACE_##TYPE                         \
	{                                                                    \
		RTTR_REGISTRATION                                                \
		{                                                                \
			using CURRENT_TYPE = TYPE;                                   \
			auto reg           = rttr::registration::class_<TYPE>(#TYPE) \
			               .constructor<>(__VA_ARGS__)(rttr::policy::ctor::as_object);

#define REFLECTION_PROPERTY(PROPERTY)                 \
	reg.property(#PROPERTY, &CURRENT_TYPE::PROPERTY); \
	SERIALIZER_REGISTER_TYPE(CURRENT_TYPE::PROPERTY)

#define REFLECTION_PROPERTY_META(PROPERTY, ...)                    \
	reg.property(#PROPERTY, &CURRENT_TYPE::PROPERTY)(__VA_ARGS__); \
	SERIALIZER_REGISTER_TYPE(CURRENT_TYPE::PROPERTY)

#define REFLECTION_END()                                                    \
	TSerializer<OutputArchive>::GetInstance().RegisterType<CURRENT_TYPE>(); \
	TSerializer<InputArchive>::GetInstance().RegisterType<CURRENT_TYPE>();  \
	}                                                                       \
	}

#ifdef NDEBUG
#	define ASSERT(x)
#else
#	define ASSERT(x) assert(x)
#endif        // NDBUG
