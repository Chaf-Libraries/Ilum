#include "MetaUtils.hpp"

namespace Ilum
{
std::string MetaUtils::GetQualifiedName(const CursorType &type)
{
	return std::string();
}

std::string MetaUtils::GetQualifiedName(const std::string &display_name, const Namespace &current_namespace)
{
	return std::string();
}

std::string MetaUtils::GetQualifiedName(const Cursor &cursor, const Namespace &current_namespace)
{
	return std::string();
}

std::string MetaUtils::FormatQualifiedName(std::string &src_string)
{
	return std::string();
}

std::vector<std::string> MetaUtils::Split(const std::string &input, const std::string &token)
{
	std::vector<std::string> result;

	std::string tmp = input;

	while (true)
	{
		size_t      index    = tmp.find(token);
		std::string sub_list = tmp.substr(0, index);
		if (!sub_list.empty())
		{
			result.push_back(sub_list);
		}
		tmp.erase(0, index + token.size());
		if (index == -1)
		{
			break;
		}
	}
	return result;
}

std::string MetaUtils::Trim(const std::string &input, const std::string &trim)
{
	std::string source   = input;
	size_t      left_pos = source.find_first_not_of(trim);
	if (left_pos == std::string::npos)
	{
		source = std::string();
	}
	else
	{
		size_t right_pos = source.find_last_not_of(trim);
		source           = source.substr(left_pos, right_pos - left_pos + 1);
	}
	return source;
}
}        // namespace Ilum