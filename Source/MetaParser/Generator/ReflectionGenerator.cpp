#include "ReflectionGenerator.hpp"
#include "Type/Field.hpp"

#include <mustache.hpp>

#include <filesystem>
#include <fstream>

namespace Ilum
{
bool ReflectionGenerator::Generate(const std::string &path, const std::string &output_path, const SchemaModule &schema)
{
	std::ifstream file;

	file.open(std::string(PROJECT_SOURCE_DIR) + "/Asset/MustacheTemplate/Reflection.mustache", std::ios::in);

	if (!file.is_open())
	{
		std::cout << std::filesystem::current_path();
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
		{
			kainjow::mustache::data header{kainjow::mustache::data::type::object};
			header["Include"] = "#include \"" + path + "\"";
			headers << header;
		}
		{
			kainjow::mustache::data header{kainjow::mustache::data::type::object};
			header["Include"] = "#include <rttr/registration.h>";
			headers << header;
		}
		reflection_data.set("Header", headers);
	}

	// Enumeration
	{
		kainjow::mustache::data enums{kainjow::mustache::data::type::list};

		for (auto &enum_ : schema.enums)
		{
			if (enum_->ShouldReflection())
			{
				enums << enum_->GenerateReflection();
			}
		}

		reflection_data.set("Enum", enums);
	}

	// Class
	{
		kainjow::mustache::data classes{kainjow::mustache::data::type::list};
		kainjow::mustache::data class_serializations{kainjow::mustache::data::type::list};
		kainjow::mustache::data field_serializations{kainjow::mustache::data::type::list};
		kainjow::mustache::data serialization_list{kainjow::mustache::data::type::list};

		for (auto &class_ : schema.classes)
		{
			if (class_->ShouldReflection())
			{
				classes << class_->GenerateReflection();
			}
			if (class_->ShouldSerialization())
			{
				{
					kainjow::mustache::data class_serialization{kainjow::mustache::data::type::object};
					class_serialization["ClassQualifiedName"] = class_->GetQualifiedName();
					class_serializations << class_serialization;
				}

				kainjow::mustache::data serialization{kainjow::mustache::data::type::object};

				bool first = true;
				for (auto &field : class_->GetFields())
				{
					kainjow::mustache::data field_serialization{kainjow::mustache::data::type::object};
					if (field->ShouldSerialization())
					{
						field_serialization["FieldQualifiedName"] = field->GetQualifiedName();
						field_serialization << field_serialization;

						serialization["IsFirst"] = first;

						first = false;
					}
				}
			}
		}

		reflection_data.set("Class", classes);
		reflection_data.set("ClassSerialization", class_serializations);
		reflection_data.set("FieldSerialization", field_serializations);
		reflection_data.set("Serialization", serialization_list);
	}

	auto result = tmpl.render(reflection_data);

	std::ofstream output(output_path);
	output.write(std::string(result).data(), result.size());
	output.flush();
	output.close();

	m_paths.push_back(std::filesystem::u8path(output_path).filename().generic_string());

	return true;
}

void ReflectionGenerator::OutputFile(const std::string &path)
{
	std::ofstream output(path);
	output << "#pragma once" << std::endl;
	for (auto &path : m_paths)
	{
		output << "#include \"_Generate/" + path + "\"" << std::endl;
	}
	output.flush();
	output.close();
}
}        // namespace Ilum