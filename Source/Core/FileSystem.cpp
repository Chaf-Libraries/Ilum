#include "FileSystem.hpp"

#include <filesystem>
#include <fstream>

namespace Ilum::Core
{
bool FileSystem::IsExist(const std::string &path)
{
	if (std::filesystem::exists(path))
	{
		return true;
	}

	return false;
}

bool FileSystem::IsFile(const std::string &path)
{
	if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path))
	{
		return true;
	}

	return false;
}

bool FileSystem::IsDirectory(const std::string &path)
{
	if (std::filesystem::exists(path) && std::filesystem::is_directory(path))
	{
		return true;
	}

	return false;
}

bool FileSystem::CreatePath(const std::string &path)
{
	if (std::filesystem::exists(path) || std::filesystem::create_directories(path))
	{
		return true;
	}

	return false;
}

bool FileSystem::DeletePath(const std::string &path)
{
	if (std::filesystem::exists(path) || std::filesystem::remove_all(path))
	{
		return true;
	}

	return false;
}

bool FileSystem::Copy(const std::string &src, const std::string &dst)
{
	if (src == dst)
	{
		return true;
	}

	if (!IsExist(GetFileDirectory(dst)))
	{
		CreatePath(GetFileDirectory(dst));
	}

	return std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
}

const std::string FileSystem::GetFileName(const std::string &path, bool extension)
{
	auto filename = std::filesystem::path(path).filename().generic_string();

	if (extension)
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

const std::string FileSystem::GetFileDirectory(const std::string &path)
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

const std::string FileSystem::GetFileExtension(const std::string &path)
{
	if (IsFile(path))
	{
		return std::filesystem::path(path).extension().generic_string();
	}

	return "";
}

const std::string FileSystem::GetRelativePath(const std::string &path)
{
	if (!IsExist(path))
	{
		return "";
	}

	return std::filesystem::relative(path, std::filesystem::current_path()).u8string();
}

const size_t FileSystem::GetFileSize(const std::string &path)
{
	if (!IsFile(path))
	{
		return 0;
	}

	return std::filesystem::file_size(path);
}

bool FileSystem::Save(const std::string &path, const std::vector<uint8_t> &data, bool binary)
{
	std::ofstream output(path, binary ? std::ofstream::binary : 2);
	output.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(uint8_t));
	output.flush();
	output.close();
	return true;
}

bool FileSystem::Read(const std::string &path, std::vector<uint8_t> &data, bool binary, uint32_t begin, uint32_t end)
{
	if (!IsFile(path))
	{
		return false;
	}

	data.clear();

	std::ifstream file;

	file.open(path, std::ios::in | (binary ? std::ios::binary : 0));

	if (!file.is_open())
	{
		return false;
	}

	uint64_t read_count = end - begin;

	// TODO: potential bug here
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
}        // namespace Ilum::Core