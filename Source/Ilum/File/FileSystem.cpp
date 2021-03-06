#include "FileSystem.hpp"

#include <filesystem>
#include <fstream>

namespace Ilum
{
bool FileSystem::isExist(const std::string &path)
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

bool FileSystem::isFile(const std::string &path)
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

bool FileSystem::isDirectory(const std::string &path)
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

bool FileSystem::createPath(const std::string &path)
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

bool FileSystem::deletePath(const std::string &path)
{
	try
	{
		if (std::filesystem::exists(path) || std::filesystem::remove_all(path))
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

bool FileSystem::copy(const std::string &src, const std::string &dst)
{
	if (src == dst)
	{
		return true;
	}
	if (!isExist(getFileDirectory(dst)))
	{
		createPath(getFileDirectory(dst));
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

const std::string FileSystem::getFileName(const std::string &path, bool has_extension)
{
	auto filename = std::filesystem::path(path).filename().generic_string();

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

		return "";
	}
}

const std::string FileSystem::getFileDirectory(const std::string &path)
{
	if (isDirectory(path))
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

const std::string FileSystem::getFileExtension(const std::string &path)
{
	try
	{
		return std::filesystem::path(path).extension().generic_string();
	}
	catch (std::system_error &e)
	{
		LOG_WARN("Failed: %s", e.what());
	}
	return "";
}

const std::string FileSystem::getRelativePath(const std::string &path)
{
	if (!isExist(path))
	{
		return "";
	}

	return std::filesystem::relative(path, std::filesystem::current_path()).u8string();
}

bool FileSystem::save(const std::string &path, const std::vector<uint8_t> &data, bool binary)
{
	std::ofstream output(path, binary ? std::ofstream::binary : 2);
	output.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(uint8_t));
	output.flush();
	output.close();
	return true;
}

bool FileSystem::read(const std::string &path, std::vector<uint8_t> &data, bool binary, uint32_t begin, uint32_t end)
{
	if (!isFile(path))
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

std::string FileSystem::toupper(const std::string &str)
{
	std::locale loc;
	std::string upper;
	for (const auto &word : str)
	{
		upper += std::toupper(word, loc);
	}

	return upper;
}

std::vector<std::string> FileSystem::split(const std::string &str, char delim)
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