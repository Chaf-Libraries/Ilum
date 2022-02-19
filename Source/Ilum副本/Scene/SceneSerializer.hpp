#pragma once

#include "File/Serializer.hpp"

#include "Scene.hpp"

namespace Ilum
{
class SceneSerializer : public Serializer
{
  public:
	virtual void serialize(const std::string &file_path) override;

	virtual void deserialize(const std::string &file_path) override;

  private:
	std::unordered_map<uint32_t, entt::entity> m_entity_lut;
};
}        // namespace Ilum