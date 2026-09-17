/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "v2d_image_ops.h"
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
extern "C" {
#include "v2d/v2d_api.h"
}

namespace vision_spacemit
{
namespace
{
class JobLock
{
public:
    JobLock()
    {
        constexpr const char* path = "/tmp/media_engine_v2d.lock";
        constexpr int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK;
        // Linux flock on local files does not require a writable descriptor.
        // Never unlink/replace this file: all cooperating processes must lock
        // the same inode, including a file previously created by root.
        for (int attempt = 0; attempt < 4; ++attempt) {
            fd_ = ::open(path, flags);
            if (fd_ >= 0 || errno != ENOENT) break;
            fd_ = ::open(path, flags | O_CREAT | O_EXCL, 0644);
            if (fd_ >= 0) {
                // Only adjust a file we exclusively created, not someone
                // else's existing lock. Remain readable with a restrictive umask.
                if (::fchmod(fd_, 0644)) fail("create permissions", errno);
                break;
            }
            if (errno != EEXIST) break;
        }
        if (fd_ < 0) fail("open", errno);
        struct stat info{};
        if (::fstat(fd_, &info)) fail("stat", errno);
        if (!S_ISREG(info.st_mode)) fail("not a regular file", EINVAL);
        int rc;
        do {
            rc = ::flock(fd_, LOCK_EX);
        } while (rc && errno == EINTR);
        if (rc) fail("acquire", errno);
    }
    ~JobLock()
    {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
    }

private:
    [[noreturn]] void fail(const char* operation, int error)
    {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        throw V2dUnavailable(std::string("V2D job lock ") + operation +
                                " failed: " + std::strerror(error));
    }
    int fd_ = -1;
};
}  // namespace
void resize_nv12_to_rgb(const cv::Mat& nv12, int input_fd, DmaBuffer& output, int width,
                        int height, int output_stride)
{
    int w = nv12.cols, h = nv12.rows * 2 / 3;
    size_t ys = nv12.step[0] * h;
    VideoFrameInfo src{}, dst{};
    src.eFrameType = dst.eFrameType = FRAME_TYPE_COMMON;
    src.stCommFrameInfo.u32Width = w;
    src.stCommFrameInfo.u32Height = h;
    src.stCommFrameInfo.ePixelFormat = MPP_PIXEL_FORMAT_NV12;
    src.stVFrame.u32PlaneNum = 2;
    src.stVFrame.u32TotalSize = ys * 3 / 2;
    for (int c = 0; c < 2; ++c) {
        src.stVFrame.u32Fd[c] = input_fd;
        src.stVFrame.u32PlaneStride[c] = nv12.step[0];
        src.stVFrame.u32PlaneSize[c] = src.stVFrame.u32PlaneSizeValid[c] =
            c ? ys / 2 : ys;
        src.stVFrame.ulPlaneVirAddr[c] = (UL)nv12.data + (c ? ys : 0);
    }
    dst.stCommFrameInfo.u32Width = width;
    dst.stCommFrameInfo.u32Height = height;
    dst.stCommFrameInfo.ePixelFormat = MPP_PIXEL_FORMAT_RGB_888;
    dst.stVFrame.u32PlaneNum = 1;
    dst.stVFrame.u32Fd[0] = output.fd();
    dst.stVFrame.ulPlaneVirAddr[0] = (UL)output.data();
    dst.stVFrame.u32PlaneStride[0] = output_stride;
    dst.stVFrame.u32TotalSize = output.size();
    dst.stVFrame.u32PlaneSize[0] = dst.stVFrame.u32PlaneSizeValid[0] =
        size_t(output_stride) * height;
    V2DArea sr{0, 0, (U16)w, (U16)h}, dr{0, 0, (U16)width, (U16)height};
    JobLock lock;
    V2DHandle job;
    int rc = V2D_BeginJob(&job);
    if (!rc) {
        rc = V2D_AddBitblitTask(job, &src, &sr, &dst, &dr,
                                V2D_CSC_MODE_BT601NARROW_2_RGB);
        if (rc)
            V2D_CancelJob(job);
        else
            rc = V2D_EndJob(job);
    }
    // MPP opens the device before submitting any task. EndJob releases its
    // job context even on this error, so there is no pending hardware work.
    if (rc == V2D_ERR_DEV_OPEN)
        throw V2dUnavailable(
            "V2D cannot open /dev/v2d_dev; check device permissions and driver");
    if (rc) throw std::runtime_error("V2D bitblit failed: " + std::to_string(rc));
}
}  // namespace vision_spacemit
