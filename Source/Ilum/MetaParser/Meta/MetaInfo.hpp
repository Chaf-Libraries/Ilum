#pragma once

#include "Parser/Cursor.hpp"

namespace Ilum
{
class MetaInfo
{
  public:
	MetaInfo(const Cursor &cursor);

	std::string GetProperty(const std::string &key) const;

	bool GetFlag(const std::string &key) const;

	const std::unordered_map<std::string, std::string> &GetProperties() const;

	kainjow::mustache::data GenerateReflection() const;

  private:
	using Property = std::pair<std::string, std::string>;
	std::vector<Property> ExtractProperties(const Cursor &cursor) const;

  private:
	std::unordered_map<std::string, std::string> m_properties;
};
}        // namespace Ilum