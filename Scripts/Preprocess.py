import sys
import os

serialize_struct = []

def read_file(file_name):
    f = open(file_name, "r")
    file = f.read()
    f.close()
    return file

def write_file(file_name, source):
    f = open(file_name, "w")
    f.write(source)
    f.close()

def append_file(file_name, source):
    f = open(file_name, "a")
    f.write(source)
    f.close()

def find_scope(source, start):
    brackets_count = 1
    for i in range(len(source) - start):
        if source[start + i] == '{':
            brackets_count += 1
        elif source[start + i] == '}':
            brackets_count -=1
        if brackets_count == 0:
            return start + i
    return -1

def parse_enum_scope(source : str):
    scope_idx = 0
    result = ""
    while True:
        scope_idx = source.find("REFLECTION_ENUM", scope_idx)
        if scope_idx == -1:
            break

        start = source.find("{", scope_idx)
        enum_name = source[scope_idx : start].replace("\n","").split(' ')[1]
        end = find_scope(source, start + 1)

        scope_idx = end

        values = source[start + 1 : end].replace(" ","").split("\n")
        for value in values:
            if value[0:2] == "//":
                values.remove(value)
        for i in range(len(values)):
            values[i] = values[i].replace(",", "")
            if len(values[i].split('=')) > 0:
                values[i] = values[i].split('=')[0]
        for value in values:
            if value == "":
                values.remove(value)
        
        result += "rttr::registration::enumeration<{}>(\"{}\")(\n".format(enum_name, enum_name)
        i = 0
        for value in values:
            result += "rttr::value(\"{}\", {}::{})".format(value, enum_name, value)
            if i < len(values) - 1:
                result += ", "
                i += 1
            result += "\n"
        result +=");\n\n"
    return result

def parse_struct_scope(source : str):
    scope_idx = 0
    result = ""
    while True:
        scope_idx = source.find("REFLECTION_STRUCT", scope_idx)
        if scope_idx == -1:
            break
        start = source.find("{", scope_idx)
        struct_name = source[scope_idx : start].replace("\n","").split(' ')[1]
        end = find_scope(source, start + 1)

        serialize_struct.append(struct_name)

        scope_idx = end

        source_props = source[start + 1 : end].replace("\n", "").replace("\t", "").split(";")

        props = []
        for prop in source_props:
            if "REFLECTION_PROPERTY" in prop:
                props.append(prop)

        struct_values = []

        result += "rttr::registration::class_<{}>(\"{}\").constructor<>()(rttr::policy::ctor::as_object)\n".format(struct_name, struct_name)

        for prop in props:
            key = prop[prop.find("(") + 1 : prop.find(")")]
            value = prop[prop.find(")") : ].split(' ')[-1]
            struct_values.append(value)
            result += ".property(\"{}\", &{}::{})\n".format(value, struct_name, value)
        result += ";\n\n"

        for value in struct_values:
            result += "SERIALIZER_REGISTER_TYPE({}::{})\n".format(struct_name, value)

        result += "\n"

    return result

if __name__ == '__main__':
    output_name = sys.argv[1]
    file_names = sys.argv[2:]
    path_name = os.path.dirname(output_name)
    entry_name = path_name + "/Generate.hpp"

    print(path_name)

    result = "#pragma once\n"
    result += "#include <Core/Macro.hpp>\n"


    header = ""
    registeration = ""

    for file_name in file_names:
        # Parsing enum
        source = read_file(file_name)
        enum_scope = parse_enum_scope(source)
        if len(enum_scope) > 0:
            header += "#include <{}>\n".format(file_name)
            registeration += enum_scope

        # Parsing class & struct
        struct_scope = parse_struct_scope(source)
        if len(struct_scope) > 0:
            header += "#include <{}>\n".format(file_name)
            registeration += struct_scope

    # serialization declaration
    serialization = ""
    for struct in serialize_struct:
        serialization += "SERIALIZER_DECLARATION({})\n".format(struct)

    if not os.path.exists(path_name):
        os.mkdir(path_name)

    result += header
    result += "namespace Ilum\n{\n"
    result += serialization
    result += "namespace Ilum_{}\n".format(abs(hash(registeration)))
    result += "{\n"
    result += "RTTR_REGISTRATION\n{\n"
    result += registeration
    result += "}\n}\n}\n"

    write_file(output_name, result)

    append_info = "#include \"{}\"\n".format(output_name)
    if os.path.exists(entry_name):
        entry = read_file(entry_name)

    entry = append_file(entry_name, append_info)
        
  