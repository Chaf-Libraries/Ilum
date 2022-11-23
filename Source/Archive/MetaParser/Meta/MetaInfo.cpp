#include "MetaInfo.hpp"
#include "MetaUtils.hpp"

namespace Ilum
{
MetaInfo::MetaInfo(const Cursor &cursor)
{
	for (auto &child : cursor.GetChildren())
	{
		if (child.GetKind() != CXCursor_AnnotateAttr)
		{
			continue;
		}
		for (auto &prop : ExtractProperties(child))
		{
			m_properties.emplace(prop);
		}
	}
}

std::string MetaInfo::GetProperty(const std::string &key) const
{
	return m_properties.find(key) == m_properties.end() ? "" : m_properties.at(key);
}

bool MetaInfo::GetFlag(const std::string &key) const
{
	return m_properties.find(key) != m_properties.end();
}

bool MetaInfo::Empty() const
{
	std::unordered_map<std::string, std::string> properties = m_properties;

	for (auto iter = properties.begin(); iter != properties.end();)
	{
		if (iter->second.empty())
		{
			iter = properties.erase(iter);
		}
		else
		{
			iter++;
		}
	}
	return properties.empty();
}

const std::unordered_map<std::string, std::string> &MetaInfo::GetProperties() const
{
	return m_properties;
}

kainjow::mustache::data MetaInfo::GenerateReflection() const
{
	kainjow::mustache::data meta_data{kainjow::mustache::data::type::object};
	if (m_properties.empty())
	{
		meta_data["Meta"] = false;
		return meta_data;
	}

	kainjow::mustache::data metas{kainjow::mustache::data::type::list};

	size_t count = 0;

	std::unordered_map<std::string, std::string> properties = m_properties;

	for (auto iter = properties.begin(); iter != properties.end();)
	{
		if (iter->second.empty())
		{
			iter = properties.erase(iter);
		}
		else
		{
			iter++;
		}
	}

	for (auto &[key, value] : properties)
	{
		count++;

		kainjow::mustache::data meta{kainjow::mustache::data::type::object};

		if (!value.empty())
		{
		}
		meta["Key"]    = key;
		meta["Value"]  = value;
		meta["IsLast"] = count == properties.size();
		metas << meta;
	}

	meta_data.set("MetaData", metas);
	meta_data["Meta"] = true;

	return meta_data;
}

std::vector<MetaInfo::Property> MetaInfo::ExtractProperties(const Cursor &cursor) const
{
	std::vector<Property> result;

	auto &&properties = MetaUtils::Split(cursor.GetDisplayName(), ",");

	static const std::string white_space_string = " \t\r\n";

	for (auto &property_item : properties)
	{
		auto &&item_details = MetaUtils::Split(property_item, "(");

		size_t left_bracket  = property_item.find_first_of("(");
		size_t right_bracket = property_item.find_last_of(")");

		auto &&key   = property_item.substr(0, left_bracket);
		auto &&value = property_item.substr(left_bracket + 1, left_bracket < right_bracket ? right_bracket - left_bracket - 1 : 0);

		key = MetaUtils::Trim(key, white_space_string);
		if (key.empty())
		{
			continue;
		}

		result.emplace_back(key, value.empty() ? "" : MetaUtils::Trim(value, white_space_string));
	}

	return result;
}
}        // namespace Ilum