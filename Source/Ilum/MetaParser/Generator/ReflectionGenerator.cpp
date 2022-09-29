#include "ReflectionGenerator.hpp"
#include "Type/Field.hpp"

#include <mustache.hpp>

#include <filesystem>
#include <fstream>

namespace Ilum
{
bool ReflectionGenerator::Generate(const std::string &path, const SchemaModule &schema)
{
	std::ifstream file;

	file.open("E:/Workspace/Ilum/Asset/MustacheTemplate/Reflection.mustache", std::ios::in);

	if (!file.is_open())
	{
		return false;
	}

	file.seekg(0, std::ios::end);
	uint64_t read_count = static_cast<uint64_t>(file.tellg());
	file.seekg(0, std::ios::beg);

	std::string data;
	data.resize(static_cast<size_t>(read_count));
	file.read(reinterpret_cast<char *>(data.data()), read_count);
	file.close();

	kainjow::mustache::mustache tmpl = {data};

	kainjow::mustache::data reflection_data{kainjow::mustache::data::type::object};
	reflection_data["Hash"] = std::to_string(std::hash<std::string>()(path));

	// Headers
	{
		kainjow::mustache::data headers{kainjow::mustache::data::type::list};
		kainjow::mustache::data header{kainjow::mustache::data::type::object};
		header["HeaderPath"] = path;
		headers << header;
		reflection_data.set("Header", headers);
	}

	// Enumeration
	{
		kainjow::mustache::data enums{kainjow::mustache::data::type::list};

		for (auto &enum_ : schema.enums)
		{
			enums << enum_->GenerateReflection();
		}

		reflection_data.set("Enum", enums);
	}

	// Class
	{
		kainjow::mustache::data classes{kainjow::mustache::data::type::list};

		for (auto &class_ : schema.classes)
		{
			classes << class_->GenerateReflection();
		}

		reflection_data.set("Class", classes);
	}

	//// Class
	//{
	//	kainjow::mustache::data classes{kainjow::mustache::data::type::list};

	//	for (auto &class_ : schema.classes)
	//	{
	//		kainjow::mustache::data class_data{kainjow::mustache::data::type::object};
	//		class_data["ClassQualifiedName"] = class_->GetQualifiedName();
	//		class_data["ClassName"]          = class_->GetName();

	//		// Fields
	//		{
	//			kainjow::mustache::data fields{kainjow::mustache::data::type::list};

	//			for (auto &field : class_->GetFields())
	//			{
	//				kainjow::mustache::data field_data{kainjow::mustache::data::type::object};
	//				field_data["FieldName"] = field->GetName();
	//				field_data["FieldQualifiedName"] = field->GetQualifiedName();
	//				kainjow::mustache::data meta_data{kainjow::mustache::data::type::object};
	//				if (!field->GetMetaData().GetProperties().empty())
	//				{
	//					meta_data["MetaBegin"]        = true;
	//					size_t                  count = 0;
	//					kainjow::mustache::data metas{kainjow::mustache::data::type::list};
	//					for (auto& [key, value] : field->GetMetaData().GetProperties())
	//					{
	//						kainjow::mustache::data meta{kainjow::mustache::data::type::object};
	//						meta["Key"] = key;
	//						meta["Value"] = value;
	//						meta["IsLast"] = count == field->GetMetaData().GetProperties().size() - 1;
	//						metas << meta;
	//					}
	//					meta_data.set("Meta", metas);
	//					meta_data["MetaEnd"] = true;
	//				}
	//				fields << field_data;
	//			}
	//			class_data.set("Fields", fields);
	//		}
	//		classes << class_data;
	//	}

	//	reflection_data.set("Class", classes);
	//}

	auto filename = std::filesystem::u8path(path).filename().generic_string();
	filename      = "Source/_Generate/" + filename.substr(0, filename.find_last_of('.')) + ".reflection.hpp";

	auto result = tmpl.render(reflection_data);

	std::ofstream output(filename);
	output.write(std::string(result).data(), result.size());
	output.flush();
	output.close();

	m_paths.push_back(filename);

	return true;
}

void ReflectionGenerator::Finish()
{
	std::ofstream output("Source/_Generate/reflection.hpp");
	output << "#pragma once" << std::endl;
	for (auto &path : m_paths)
	{
		output << "#include \"" + path + "\"" << std::endl;
	}
	output.flush();
	output.close();
}
}        // namespace Ilum