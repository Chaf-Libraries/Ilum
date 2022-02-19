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

template <>
struct convert<glm::mat4>
{
	static Node encode(const glm::mat4 &rhs)
	{
		Node node;
		node.push_back(rhs[0][0]);
		node.push_back(rhs[0][1]);
		node.push_back(rhs[0][2]);
		node.push_back(rhs[0][3]);
		node.push_back(rhs[1][0]);
		node.push_back(rhs[1][1]);
		node.push_back(rhs[1][2]);
		node.push_back(rhs[1][3]);
		node.push_back(rhs[2][0]);
		node.push_back(rhs[2][1]);
		node.push_back(rhs[2][2]);
		node.push_back(rhs[2][3]);
		node.push_back(rhs[3][0]);
		node.push_back(rhs[3][1]);
		node.push_back(rhs[3][2]);
		node.push_back(rhs[3][3]);
		node.SetStyle(EmitterStyle::Flow);
		return node;
	}

	static bool decode(const Node &node, glm::mat4 &rhs)
	{
		if (!node.IsSequence() || node.size() != 16)
			return false;

		rhs[0][0] = node[0].as<float>();
		rhs[0][1] = node[1].as<float>();
		rhs[0][2] = node[2].as<float>();
		rhs[0][3] = node[3].as<float>();
		rhs[1][0] = node[4].as<float>();
		rhs[1][1] = node[5].as<float>();
		rhs[1][2] = node[6].as<float>();
		rhs[1][3] = node[7].as<float>();
		rhs[2][0] = node[8].as<float>();
		rhs[2][1] = node[9].as<float>();
		rhs[2][2] = node[10].as<float>();
		rhs[2][3] = node[11].as<float>();
		rhs[3][0] = node[12].as<float>();
		rhs[3][1] = node[13].as<float>();
		rhs[3][2] = node[14].as<float>();
		rhs[3][3] = node[15].as<float>();
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

inline YAML::Emitter &operator<<(YAML::Emitter &emitter, const glm::mat4 &m)
{
	emitter << YAML::Flow;
	emitter << YAML::BeginSeq << m[0][0] << m[0][1] << m[0][2] << m[0][3] << m[1][0] << m[1][1] << m[1][2] << m[1][3] << m[2][0] << m[2][1] << m[2][2] << m[2][3] << m[3][0] << m[3][1] << m[3][2] << m[3][3] << YAML::EndSeq;
	return emitter;
}

class Serializer
{
  public:
	Serializer() = default;

	~Serializer() = default;

	virtual void serialize(const std::string &file_path) = 0;

	virtual void deserialize(const std::string &file_path) = 0;
};
}        // namespace Ilum