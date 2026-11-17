#include "strata/core/latent_dense.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {

#if defined(_WIN32)
// Windows: CreateFileA + CreateFileMappingA + MapViewOfFile.  Same contract as the POSIX path:
// strict size check, read-only mapping, unmap on close.  The mapping handle keeps the section
// alive after the file handle closes.
static void win_err(std::string& err, const std::string& path, const char* what) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), " (win32 error %lu)", (unsigned long) GetLastError());
    err = path + ": " + what + buf;
}
#endif

LatentDense::~LatentDense() { close(); }

void LatentDense::close() {
#if defined(_WIN32)
    if (base_) UnmapViewOfFile(base_);
    if (map_) CloseHandle(map_);
#else
    if (base_) munmap(base_, size_);
#endif
    base_ = nullptr; map_ = nullptr;
    size_ = 0; bytes_ = 0;
}

bool LatentDense::open(const std::string& path, int64_t n_layers, int64_t n_groups, int64_t n_embd,
                       int64_t latent_dim, std::string& err) {
    close();
    n_layers_ = n_layers; n_groups_ = n_groups; n_embd_ = n_embd; latent_dim_ = latent_dim;
    // Q4DU: the table's FORMAT IS A PROPERTY OF THE FILE.  STRATA_LATENT_DENSE_FMT=4 or =8 still forces it (and a
    // forced format whose size disagrees stays a hard error - a misread layout would produce garbage answers), but
    // when the variable is unset the file's OWN SIZE decides: 18 B a 32-weight block is Q4_0, 34 B is Q8_0, both
    // from the same geometry.  A switch this cheap to get wrong must not depend on an env var someone remembers to
    // set - the Q4 pack is otherwise indistinguishable from "the fast tiled kernels silently did not run".
    const bool fmt_forced = ::getenv("STRATA_LATENT_DENSE_FMT") != nullptr;
    fmt_ = 8;
    if (const char* f = ::getenv("STRATA_LATENT_DENSE_FMT")) {
        if (f[0] == '4' && f[1] == 0) fmt_ = 4;
        else if (f[0] == '8' && f[1] == 0) fmt_ = 8;
        else { err = path + ": STRATA_LATENT_DENSE_FMT must be 4 or 8 (got '" + std::string(f) + "')"; return false; }
    }
    const int env_fmt = fmt_;          // the env's choice, captured BEFORE want_for() moves fmt_ around
    auto want_for = [&](int f) -> uint64_t {
        fmt_ = f;
        return (uint64_t) n_layers * (uint64_t) (d_bytes() + u_bytes());
    };
    const uint64_t want8 = want_for(8), want4 = want_for(4);
    fmt_ = env_fmt;
    uint64_t want = (fmt_ == 4) ? want4 : want8;
    auto resolve_fmt = [&](uint64_t size) -> bool {
        want = (fmt_ == 4) ? want4 : want8;
        if (size == want) return true;
        const int other = (fmt_ == 4) ? 8 : 4;
        const uint64_t alt = (other == 4) ? want4 : want8;
        if (!fmt_forced && size == alt) {
            fmt_ = other; want = alt;
            std::fprintf(stderr, "latent dense: %s is the %s pack (%llu B); fmt %d selected from its size\n",
                         path.c_str(), other == 4 ? "Q4_0" : "Q8_0", (unsigned long long) size, other);
            return true;
        }
        err = path + ": size " + std::to_string((long long) size) + " != " + std::to_string(want) +
              " (fmt " + std::to_string(fmt_) + " wants " + std::to_string(want) +
              "; fmt " + std::to_string(other) + " wants " + std::to_string((long long) alt) + ")";
        return false;
    };
#if defined(_WIN32)
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) { win_err(err, path, "CreateFile"); return false; }
    LARGE_INTEGER fsz;
    if (!GetFileSizeEx(fh, &fsz)) { win_err(err, path, "GetFileSizeEx"); CloseHandle(fh); return false; }
    if (!resolve_fmt((uint64_t) fsz.QuadPart)) {
        CloseHandle(fh); return false;
    }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) { win_err(err, path, "CreateFileMapping"); CloseHandle(fh); return false; }
    CloseHandle(fh);   // the mapping object holds the section
    void* p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!p) { win_err(err, path, "MapViewOfFile"); CloseHandle(mh); return false; }
    map_ = mh;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = path + ": " + std::strerror(errno); return false; }
    struct stat st;
    if (fstat(fd, &st) != 0) { err = path + ": fstat: " + std::strerror(errno); ::close(fd); return false; }
    if (!resolve_fmt((uint64_t) st.st_size)) {   // strict: a forced format and the file's size must agree.
        ::close(fd); return false;
    }
    void* p = mmap(nullptr, (size_t) want, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) { err = path + ": mmap: " + std::strerror(errno); return false; }
#endif
    base_ = (uint8_t*) p; size_ = (size_t) want; bytes_ = want;
    return true;
}

const uint8_t* LatentDense::d(int64_t layer) const {
    if (!base_ || layer < 0 || layer >= n_layers_) return nullptr;
    return base_ + (size_t) layer * (d_bytes() + u_bytes());
}

const uint8_t* LatentDense::u(int64_t layer) const {
    if (!base_ || layer < 0 || layer >= n_layers_) return nullptr;
    return d(layer) + d_bytes();
}

LatentA::~LatentA() { close(); }
void LatentA::close() {
#if defined(_WIN32)
    if (base_) UnmapViewOfFile(base_);
    if (map_) CloseHandle(map_);
#else
    if (base_) munmap(base_, size_);
#endif
    base_ = nullptr; map_ = nullptr; size_ = 0;
}
bool LatentA::open(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t blob_bytes, std::string& err) {
    close();
    n_layers_ = n_layers; n_expert_ = n_expert; blob_bytes_ = blob_bytes;
    const uint64_t want = (uint64_t) n_layers * (uint64_t) n_expert * (uint64_t) blob_bytes;
#if defined(_WIN32)
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) { win_err(err, path, "CreateFile"); return false; }
    LARGE_INTEGER fsz;
    if (!GetFileSizeEx(fh, &fsz)) { win_err(err, path, "GetFileSizeEx"); CloseHandle(fh); return false; }
    if ((uint64_t) fsz.QuadPart != want) {
        err = path + ": size " + std::to_string((long long) fsz.QuadPart) + " != " + std::to_string(want);
        CloseHandle(fh); return false;
    }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) { win_err(err, path, "CreateFileMapping"); CloseHandle(fh); return false; }
    CloseHandle(fh);
    void* p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!p) { win_err(err, path, "MapViewOfFile"); CloseHandle(mh); return false; }
    map_ = mh;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = path + ": " + std::strerror(errno); return false; }
    struct stat st;
    if (fstat(fd, &st) != 0 || (uint64_t) st.st_size != want) {
        err = path + ": size " + std::to_string((long long) st.st_size) + " != " + std::to_string(want);
        ::close(fd); return false;
    }
    void* p = mmap(nullptr, (size_t) want, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) { err = path + ": mmap: " + std::strerror(errno); return false; }
#endif
    base_ = (uint8_t*) p; size_ = (size_t) want;
    return true;
}
const uint8_t* LatentA::blob(int64_t layer, int64_t expert) const {
    if (!base_ || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return nullptr;
    return base_ + ((size_t) layer * (size_t) n_expert_ + (size_t) expert) * (size_t) blob_bytes_;
}

}  // namespace strata::core
