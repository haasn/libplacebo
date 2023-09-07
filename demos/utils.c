// License: CC0 / Public Domain

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "../src/os.h"

#ifdef PL_HAVE_WIN32
#include <shlobj.h>
#endif

#ifdef PL_HAVE_APPLE
#include <sys/types.h>
#include <pwd.h>
#endif

#ifdef PL_HAVE_UNIX
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

const char *get_cache_dir(char (*buf)[512])
{
    // Check if XDG_CACHE_HOME is set for Linux/BSD
    const char* xdg_cache_home = getenv("XDG_CACHE_HOME");
    if (xdg_cache_home)
        return xdg_cache_home;

#ifdef _WIN32
    const char* local_app_data = getenv("LOCALAPPDATA");
    if (local_app_data)
        return local_app_data;
#endif

#ifdef __APPLE__
    struct passwd* pw = getpwuid(getuid());
    if (pw) {
        int ret = snprintf(*buf, sizeof(*buf), "%s/%s", pw->pw_dir, "Library/Caches");
        if (ret > 0 && ret < sizeof(*buf))
            return *buf;
    }
#endif

    const char* home = getenv("HOME");
    if (home) {
        int ret = snprintf(*buf, sizeof(*buf), "%s/.cache", home);
        if (ret > 0 && ret < sizeof(*buf))
            return *buf;
    }

    return NULL;
}
