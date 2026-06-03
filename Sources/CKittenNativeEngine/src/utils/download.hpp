// src/utils/download.hpp — header-only weight download utility
//
// ensure_file(local_path, url):
//   If local_path does not exist, download it from url using curl.
//   Returns true on success, false on failure.
//   Creates parent directories as needed.

#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <filesystem>

inline bool ensure_file(const std::string& local_path, const std::string& url)
{
    namespace fs = std::filesystem;
    if (fs::exists(local_path)) return true;

    if (fs::path p(local_path); p.has_parent_path())
        fs::create_directories(p.parent_path());

    printf("  [download] %s\n            <- %s\n", local_path.c_str(), url.c_str());
    fflush(stdout);

    // -L: follow redirects (HuggingFace LFS uses them)
    // -f: fail on HTTP error (4xx/5xx)
    // --progress-bar: single-line progress
    std::string cmd = "curl -L -f --progress-bar -o \""
                    + local_path + "\" \"" + url + "\"";
    int ret = system(cmd.c_str());
    if (ret != 0) {
        fs::remove(local_path);   // remove partial file
        fprintf(stderr, "\n  [download FAILED] exit=%d  %s\n", ret, url.c_str());
        return false;
    }
    printf("  [download OK] %s\n", local_path.c_str());
    return true;
}
