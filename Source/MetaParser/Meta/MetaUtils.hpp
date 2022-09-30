#pragma once

#include "Parser/Cursor.hpp"
#include "Type/Namespace.hpp"

namespace Ilum
{
class MetaUtils
{
  public:
	static std::string GetQualifiedName(const CursorType &type);

	static std::string GetQualifiedName(const std::string &display_name, const Namespace &current_namespace);

	static std::string GetQualifiedName(const Cursor &cursor, const Namespace &current_namespace);

	static std::string FormatQualifiedName(std::string &src_string);

	static std::vector<std::string> Split(const std::string &input, const std::string &token);

	static std::string Trim(const std::string &input, const std::string &trim);
};
}        // namespace Ilum