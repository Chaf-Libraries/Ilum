#pragma once

#include "Singleton.hpp"

#include <filesystem>

namespace Ilum
{
	class Path :public Singleton<Path>
	{
	  public:
	    bool IsExist(const std::string &path);
	    bool IsFile(const std::string &path);
	    bool IsDirectory(const std::string &path);
	    bool CreatePath(const std::string &path);
	    bool DeletePath(const std::string &path);
	    bool Copy(const std::string &src, const std::string &dst);
	    void SetCurrent(const std::string &path);

		const std::string GetCurrent();
	    const std::string GetFileName(const std::string &path, bool has_extension = true);
	    const std::string GetFileDirectory(const std::string &path);
	    const std::string GetFileExtension(const std::string &path);
	    const std::string GetRelativePath(const std::string &path);

	    bool Save(const std::string &path, const std::vector<uint8_t> &data, bool binary = false);
	    bool Read(const std::string &path, std::vector<uint8_t> &data, bool binary = false, uint32_t begin = 0, uint32_t end = 0);

	    std::string              Toupper(const std::string &str);
	    std::vector<std::string> Split(const std::string &str, char delim);
	};
}