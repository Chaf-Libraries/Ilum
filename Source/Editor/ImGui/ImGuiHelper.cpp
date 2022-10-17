#include "ImGuiHelper.hpp"
#include "Editor/Editor.hpp"

#include <RHI/RHITexture.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <rttr/variant.h>

namespace ImGui
{
inline rttr::variant GetMetaData(const std::string &key, rttr::variant &var, const rttr::property *prop)
{
	if (prop && prop->get_metadata(key))
	{
		return prop->get_metadata(key);
	}
	else if (var && var.get_type().get_metadata(key))
	{
		return var.get_type().get_metadata(key);
	}
	return {};
}

namespace Impl
{
using MetaCallback = std::function<rttr::variant(std::string)>;

template <typename T, typename...>
struct VariantEditor
{};

template <>
struct VariantEditor<bool>
{
	inline bool operator()(const std::string &name, rttr::variant &var, const rttr::property *prop)
	{
		bool val = var.convert<bool>();
		if (ImGui::Checkbox(name.c_str(), &val))
		{
			var = val;
			return true;
		}
		return false;
	}
};

template <>
struct VariantEditor<Ilum::RHITexture *>
{
	inline bool operator()(const std::string &name, rttr::variant &var, const rttr::property *prop)
	{
		auto val = var.convert<Ilum::RHITexture *>();
		if (!val)
		{
			return false;
		}

		if (GetMetaData("Editor", var, prop) == "Button")
		{
			ImGui::ImageButton(val, ImVec2{100, 100});
		}
		else
		{
			ImGui::Image(val, ImVec2{100, 100});
		}
		return false;
	}
};

template <>
struct VariantEditor<std::string>
{
	inline bool operator()(const std::string &name, rttr::variant &var, const rttr::property *prop)
	{
		std::string str     = var.convert<std::string>();
		char        buf[64] = {0};
		std::memcpy(buf, str.data(), sizeof(buf));
		if (ImGui::InputText(name.c_str(), buf, sizeof(buf)))
		{
			str = buf;
			var = str;
			return true;
		}

		if (GetMetaData("DragDrop", var, prop))
		{
			if (ImGui::BeginDragDropTarget())
			{
				if (const auto *pay_load = ImGui::AcceptDragDropPayload(GetMetaData("DragDrop", var, prop).convert<std::string>().c_str()))
				{
					ASSERT(pay_load->DataSize == sizeof(std::string));
					str = *static_cast<std::string *>(pay_load->Data);
					var = str;
					ImGui::EndDragDropTarget();
					return true;
				}
				ImGui::EndDragDropTarget();
			}
		}
		return false;
	}
};

template <typename T, glm::length_t N>
struct VariantEditor<glm::vec<N, T>>
{
	template <typename T>
	inline ImGuiDataType GetDataType()
	{
		return ImGuiDataType_COUNT;
	}

	template <>
	inline ImGuiDataType GetDataType<float>()
	{
		return ImGuiDataType_Float;
	}

	template <>
	inline ImGuiDataType GetDataType<int32_t>()
	{
		return ImGuiDataType_S32;
	}

	template <>
	inline ImGuiDataType GetDataType<uint32_t>()
	{
		return ImGuiDataType_U32;
	}

	template <>
	inline ImGuiDataType GetDataType<size_t>()
	{
		return ImGuiDataType_U64;
	}

	template <typename T, glm::length_t N>
	bool DragScalars(const std::string &name, rttr::variant &var, ImGuiDataType data_type, T min_val, T max_val)
	{
		if (N == 1)
		{
			T val = var.convert<T>();

			if (ImGui::DragScalar(name.c_str(), data_type, &val, 0.1f, &min_val, &max_val, rttr::type::get<T>() == rttr::type::get<float>() ? "%.2f" : (rttr::type::get<T>() == rttr::type::get<size_t>() ? "%ld" : "%d")))
			{
				var = val;
				return true;
			}
		}
		else
		{
			glm::vec<N, T> val = var.convert<glm::vec<N, T>>();
			if (ImGui::DragScalarN(name.c_str(), data_type, &val.x, N, 0.1f, &min_val, &max_val, rttr::type::get<T>() == rttr::type::get<float>() ? "%.2f" : (rttr::type::get<T>() == rttr::type::get<size_t>() ? "%ld" : "%d")))
			{
				var = val;
				return true;
			}
		}
		return false;
	}

	template <typename T, glm::length_t N>
	bool ColorEdit(const std::string &name, rttr::variant &var)
	{
		ASSERT(typeid(T) == typeid(float) && (N == 3 || N == 4));
		glm::vec<N, T> val = var.convert<glm::vec<N, T>>();
		if (N == 3 && ImGui::ColorEdit3(name.c_str(), (float *) &val.x, ImGuiColorEditFlags_NoInputs))
		{
			var = val;
			return true;
		}
		else if (N == 4 && ImGui::ColorEdit4(name.c_str(), (float *) &val.x, ImGuiColorEditFlags_NoInputs))
		{
			var = val;
			return true;
		}
		return false;
	}

	template <typename T, glm::length_t N>
	bool ColorPicker(const std::string &name, rttr::variant &var)
	{
		ASSERT(typeid(T) == typeid(float) && (N == 3 || N == 4));
		glm::vec<N, T> val = var.convert<glm::vec<N, T>>();
		if (N == 3 && ImGui::ColorPicker3(name.c_str(), (float *) &val.x, ImGuiColorEditFlags_NoInputs))
		{
			var = val;
			return true;
		}
		else if (N == 4 && ImGui::ColorPicker4(name.c_str(), (float *) &val.x, ImGuiColorEditFlags_NoInputs))
		{
			var = val;
			return true;
		}
		return false;
	}

	template <typename T, glm::length_t N>
	bool SliderScalars(const std::string &name, rttr::variant &var, ImGuiDataType data_type, T min_val, T max_val)
	{
		if (N == 1)
		{
			T val = var.convert<T>();
			if (ImGui::SliderScalar(name.c_str(), data_type, &val, &min_val, &max_val, "%.2f"))
			{
				var = val;
				return true;
			}
		}
		else
		{
			glm::vec<N, T> val = var.convert<glm::vec<N, T>>();
			if (ImGui::SliderScalarN(name.c_str(), data_type, &val.x, N, &min_val, &max_val, "%.2f"))
			{
				var = val;
				return true;
			}
		}
		return false;
	}

	inline bool operator()(const std::string &name, rttr::variant &var, const rttr::property *prop)
	{
		ImGuiDataType data_type = GetDataType<T>();

		std::string display_name = GetMetaData("Name", var, prop) ? GetMetaData("Name", var, prop).to_string() : name;

		T min_val = GetMetaData("Min", var, prop) ? GetMetaData("Min", var, prop).convert<T>() : std::numeric_limits<T>::lowest();
		T max_val = GetMetaData("Max", var, prop) ? GetMetaData("Max", var, prop).convert<T>() : std::numeric_limits<T>::max();

		bool update = false;

		if (GetMetaData("Editor", var, prop) == "ColorEdit")
		{
			update = ColorEdit<T, N>(display_name, var);
		}
		else if (GetMetaData("Editor", var, prop) == "ColorPicker")
		{
			update = ColorPicker<T, N>(display_name, var);
		}
		else if (GetMetaData("Editor", var, prop) == "Slider")
		{
			update = SliderScalars<T, N>(display_name, var, data_type, min_val, max_val);
		}
		else
		{
			update = DragScalars<T, N>(display_name, var, data_type, min_val, max_val);
		}

		if (GetMetaData("DragDrop", var, prop))
		{
			if (ImGui::BeginDragDropTarget())
			{
				if (const auto *pay_load = ImGui::AcceptDragDropPayload(GetMetaData("DragDrop", var, prop).convert<std::string>().c_str()))
				{
					if (N == 1)
					{
						var = *static_cast<T *>(pay_load->Data);
					}
					else
					{
						var = *static_cast<glm::vec<N, T> *>(pay_load->Data);
					}
					update = true;
				}
				ImGui::EndDragDropTarget();
			}
		}

		return update;
	}
};

}        // namespace Impl

inline static std::unordered_map<rttr::type, std::function<bool(const std::string &, rttr::variant &, const rttr::property *)>> EditorMap = {
#define EDIT_ITEM(TYPE)                                                                                                                                                       \
	{                                                                                                                                                                         \
		rttr::type::get<TYPE>(), [](const std::string &name, rttr::variant &var, const rttr::property *prop) -> bool { return Impl::VariantEditor<TYPE>()(name, var, prop); } \
	}

    {rttr::type::get<float>(), [](const std::string &name, rttr::variant &var, const rttr::property *prop) -> bool { return Impl::VariantEditor<glm::vec1>()(name, var, prop); }},
    {rttr::type::get<int32_t>(), [](const std::string &name, rttr::variant &var, const rttr::property *prop) -> bool { return Impl::VariantEditor<glm::ivec1>()(name, var, prop); }},
    {rttr::type::get<uint32_t>(), [](const std::string &name, rttr::variant &var, const rttr::property *prop) -> bool { return Impl::VariantEditor<glm::uvec1>()(name, var, prop); }},
    {rttr::type::get<size_t>(), [](const std::string &name, rttr::variant &var, const rttr::property *prop) -> bool { return Impl::VariantEditor<glm::u64vec1>()(name, var, prop); }},

    EDIT_ITEM(bool),

    EDIT_ITEM(glm::vec1),
    EDIT_ITEM(glm::vec2),
    EDIT_ITEM(glm::vec3),
    EDIT_ITEM(glm::vec4),
    EDIT_ITEM(glm::vec4),

    EDIT_ITEM(glm::ivec1),
    EDIT_ITEM(glm::ivec2),
    EDIT_ITEM(glm::ivec3),
    EDIT_ITEM(glm::ivec4),
    EDIT_ITEM(glm::ivec4),

    EDIT_ITEM(glm::uvec1),
    EDIT_ITEM(glm::uvec2),
    EDIT_ITEM(glm::uvec3),
    EDIT_ITEM(glm::uvec4),
    EDIT_ITEM(glm::uvec4),

    EDIT_ITEM(glm::u64vec1),
    EDIT_ITEM(glm::u64vec2),
    EDIT_ITEM(glm::u64vec3),
    EDIT_ITEM(glm::u64vec4),
    EDIT_ITEM(glm::u64vec4),

    EDIT_ITEM(std::string)};

bool EditVariantImpl(const std::string &name, Ilum::Editor *editor, rttr::variant &var, const rttr::property *prop)
{
	bool update = false;

	if (GetMetaData("Editor", var, prop) == "Disable")
	{
		return false;
	}

	if (var.get_type() == rttr::type::get<Ilum::RHITexture *>())
	{
		Impl::VariantEditor<Ilum::RHITexture *>()(name, var, prop);
		return false;
	}
	else if (EditorMap.find(var.get_type()) != EditorMap.end())
	{
		update = EditorMap[var.get_type()](name, var, prop);
		if (var.get_type() == rttr::type::get<size_t>() && GetMetaData("Editor", var, prop) == "Texture")
		{
			auto          resource = editor->GetRenderer()->GetResourceManager()->GetResource<Ilum::ResourceType::Texture>(var.convert<size_t>());
			rttr::variant texture  = resource ? resource->GetTexture() : nullptr;
			Impl::VariantEditor<Ilum::RHITexture *>()(name, texture, prop);
		}
	}
	else if (var.get_type().is_enumeration())
	{
		auto enumeration = var.get_type().get_enumeration();

		std::vector<const char *> enums;
		enums.reserve(enumeration.get_values().size());

		uint64_t prop_enum        = var.convert<uint64_t>();
		int32_t  current_enum_idx = 0;

		for (auto &enum_ : enumeration.get_values())
		{
			if (prop_enum == enum_.convert<uint64_t>())
			{
				current_enum_idx = static_cast<int32_t>(enums.size());
			}
			enums.push_back(enumeration.value_to_name(enum_).data());
		}

		if (ImGui::Combo(name.c_str(), reinterpret_cast<int32_t *>(&current_enum_idx), enums.data(), static_cast<int32_t>(enums.size())))
		{
			var    = enumeration.name_to_value(enums[current_enum_idx]);
			update = true;
		}
	}
	else if (var.get_type().is_sequential_container())
	{
		auto     seq_view = var.create_sequential_view();
		uint32_t count    = 0;
		for (size_t i = 0; i < seq_view.get_size(); i++)
		{
			std::string elem_name = fmt::format("{} - {}", name, i);
			if (ImGui::TreeNode(elem_name.c_str()))
			{
				rttr::variant elem = seq_view.get_value_type().create();
				update             = EditVariantImpl(elem_name.c_str(), editor, elem, prop);
				seq_view.set_value(i, elem);
				ImGui::TreePop();
			}
		}
	}
	else if (var.get_type().is_class())
	{
		for (auto &property_ : var.get_type().get_properties())
		{
			auto prop_val = property_.get_value(var);
			update        = EditVariantImpl(property_.get_name().to_string().c_str(), editor, prop_val, &property_);
			property_.set_value(var, prop_val);
		}
	}

	return update;
}
}        // namespace ImGui