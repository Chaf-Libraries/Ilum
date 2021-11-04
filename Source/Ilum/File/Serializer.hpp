#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include <glm/glm.hpp>

namespace YAML
{
template <>
struct convert<glm::vec3>
{
	static Node encode(const glm::vec3 &rhs)
	{
		Node node;
		node.push_back(rhs.x);
		node.push_back(rhs.y);
		node.push_back(rhs.z);
		node.SetStyle(EmitterStyle::Flow);
		return node;
	}

	static bool decode(const Node &node, glm::vec3 &rhs)
	{
		if (!node.IsSequence() || node.size() != 3)
			return false;

		rhs.x = node[0].as<float>();
		rhs.y = node[1].as<float>();
		rhs.z = node[2].as<float>();
		return true;
	}
};

template <>
struct convert<glm::vec4>
{
	static Node encode(const glm::vec4 &rhs)
	{
		Node node;
		node.push_back(rhs.x);
		node.push_back(rhs.y);
		node.push_back(rhs.z);
		node.push_back(rhs.w);
		node.SetStyle(EmitterStyle::Flow);
		return node;
	}

	static bool decode(const Node &node, glm::vec4 &rhs)
	{
		if (!node.IsSequence() || node.size() != 4)
			return false;

		rhs.x = node[0].as<float>();
		rhs.y = node[1].as<float>();
		rhs.z = node[2].as<float>();
		rhs.w = node[3].as<float>();
		return true;
	}
};
}        // namespace YAML

namespace Ilum
{
inline YAML::Emitter &operator<<(YAML::Emitter &emitter, const glm::vec3 &v)
{
	emitter << YAML::Flow;
	emitter << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
	return emitter;
}

inline YAML::Emitter &operator<<(YAML::Emitter &emitter, const glm::vec4 &v)
{
	emitter << YAML::Flow;
	emitter << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
	return emitter;
}

template <typename T>
class Serializer
{
  public:
	Serializer(T &data) :
	    m_data(data)
	{
	}

	~Serializer() = default;

	virtual void serialize(const std::string &file_path) = 0;

	virtual void deserialize(const std::string &file_path) = 0;

  protected:
	T &m_data;
};
}        // namespace Ilum