#include "Path.hpp"
#include "Core.hpp"

namespace Ilum
{
Path &Path::GetInstance()
{
	static Path path;
	return path;
}

bool Path::IsExist(const std::string &path)
{
	try
	{
		if (std::filesystem::exists(path))
		{
			return true;
		}
	}
	catch (std::filesystem::filesystem_error &e)
	{
		LOG_WARN("%s. %s", e.what(), path.c_str());
	}
	return false;
}

bool Path::IsFile(const std::string &path)
{
	try
	{
		if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path))
		{
			return true;
		}
	}
	catch (std::filesystem::filesystem_error &e)
	{
		LOG_WARN("%s. %s", e.what(), path.c_str());
	}
	return false;
}

bool Path::IsDirectory(const std::string &path)
{
	try
	{
		if (std::filesystem::exists(path) && std::filesystem::is_directory(path))
		{
			return true;
		}
	}
	catch (std::filesystem::filesystem_error &e)
	{
		LOG_WARN("%s. %s", e.what(), path.c_str());
	}
	return false;
}

bool Path::CreatePath(const std::string &path)
{
	try
	{
		if (std::filesystem::exists(path) || std::filesystem::create_directories(path))
		{
			return true;
		}
	}
	catch (std::filesystem::filesystem_error &e)
	{
		LOG_WARN("%s. %s", e.what(), path.c_str());
	}
	return false;
}

bool Path::DeletePath(const std::string &path)
{
	try
	{
		if (std::filesystem::exists(path) && std::filesystem::remove_all(path))
		{
			return true;
		}
	}
	catch (std::filesystem::filesystem_error &e)
	{
		LOG_WARN("%s. %s", e.what(), path.c_str());
	}
	return false;
}

bool Path::Copy(const std::string &src, const std::string &dst)
{
	if (src == dst)
	{
		return true;
	}
	if (!IsExist(GetFileDirectory(dst)))
	{
		CreatePath(GetFileDirectory(dst));
	}
	try
	{
		return std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
	}
	catch (std::filesystem::filesystem_error &e)
	{
		LOG_WARN("%s", e.what());
	}
	return false;
}

void Path::SetCurrent(const std::string &path)
{
	std::filesystem::current_path(path);
}

const std::string Path::GetCurrent(bool convert)
{
	std::string current = std::filesystem::current_path().string();
	if (convert)
	{
		std::replace(current.begin(), current.end(), '\\', '/');
	}
	return current;
}

const std::string Path::GetFileName(const std::string &path, bool has_extension)
{
	auto filename = std::filesystem::u8path(path).filename().generic_string();

	if (has_extension)
	{
		return filename;
	}
	else
	{
		size_t last_index = filename.find_last_of('.');

		if (last_index != std::string::npos)
		{
			return filename.substr(0, last_index);
		}

		return filename;
	}
}

const std::string Path::GetFileDirectory(const std::string &path)
{
	if (IsDirectory(path))
	{
		return path;
	}

	size_t last_index = path.find_last_of("\\/");

	if (last_index != std::string::npos)
	{
		return path.substr(0, last_index + 1);
	}

	return "";
}

const std::string Path::GetFileExtension(const std::string &path)
{
	try
	{
		return std::filesystem::u8path(path).extension().generic_string();
	}
	catch (std::system_error &e)
	{
		LOG_WARN("Failed: %s", e.what());
	}
	return "";
}

const std::string Path::GetRelativePath(const std::string &path)
{
	if (!IsExist(path))
	{
		return "";
	}

	return std::filesystem::relative(path, std::filesystem::current_path()).u8string();
}

bool Path::Save(const std::string &path, const std::vector<uint8_t> &data, bool binary)
{
	std::string dir = GetFileDirectory(path);
	if (!IsExist(dir))
	{
		CreatePath(dir);
	}
	std::ofstream output(path, binary ? std::ofstream::binary : 2);
	output.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(uint8_t));
	output.flush();
	output.close();
	return true;
}

bool Path::Read(const std::string &path, std::vector<uint8_t> &data, bool binary, uint32_t begin, uint32_t end)
{
	if (!IsFile(path))
	{
		LOG_ERROR("Failed to read file {}", path);
		return false;
	}

	data.clear();

	std::ifstream file;

	file.open(path, std::ios::in | (binary ? std::ios::binary : 0));

	if (!file.is_open())
	{
		LOG_ERROR("Failed to read file {}", path);
		return false;
	}

	uint64_t read_count = end - begin;

	if ((end == begin && begin == 0) || end < begin)
	{
		file.seekg(0, std::ios::end);
		read_count = static_cast<uint64_t>(file.tellg());
		file.seekg(0, std::ios::beg);
	}

	data.resize(static_cast<size_t>(read_count));
	file.read(reinterpret_cast<char *>(data.data()), read_count);
	file.close();

	return true;
}

std::string Path::Toupper(const std::string &str)
{
	std::locale loc;
	std::string upper;
	for (const auto &word : str)
	{
		upper += std::toupper(word, loc);
	}

	return upper;
}

std::vector<std::string> Path::Split(const std::string &str, char delim)
{
	std::vector<std::string> tokens;

	std::stringstream sstream(str);
	std::string       token;
	while (std::getline(sstream, token, delim))
	{
		tokens.push_back(token);
	}

	return tokens;
}
}        // namespace Ilum