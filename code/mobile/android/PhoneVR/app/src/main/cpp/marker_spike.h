#ifndef PHONEVR_MARKER_SPIKE_H
#define PHONEVR_MARKER_SPIKE_H

// STAGE 1 SPIKE for marker-based tracking. It does NOT track anything yet. It opens the back
// camera (NDK camera API), runs the AprilTag detector on every frame it can keep up with, and
// logs once a second what it found and how long it took, so we can see whether a weak phone
// (e.g. UMIDIGI A3S) can do this at all before building the real thing on top.
//
// Logcat tag: PhoneVR_MarkerSpike. Needs the CAMERA permission, and ARCore must be off (both
// want the camera).
//
// What the log tells you
//   camera:   the back camera's hardware level, whether its image timestamps share the sensors'
//             clock (REALTIME) or not (UNKNOWN), focal length / sensor size / field of view
//   per second: frames the camera delivered, frames processed (the detector works on the
//             newest frame and skips the rest), detect time avg/max in ms, and for every tag
//             seen: id, how many frames it was seen in, mean size in pixels and decision margin
//   latency:  how old a frame already is when we get it (only meaningful if REALTIME)

#include <android/log.h>
#include <android/native_window.h>
#include <camera/NdkCameraCaptureSession.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraMetadata.h>
#include <camera/NdkCaptureRequest.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

#include "marker_detector.h"

namespace phonevr {

class MarkerSpike {
public:
    struct Config {
        int wantWidth = 640;    // closest 4:3 YUV stream to this is used
        int wantHeight = 480;
        MarkerDetector::Config detector;
    };

    MarkerSpike() : MarkerSpike(Config()) {}
    explicit MarkerSpike(const Config &cfg) : cfg_(cfg) {}
    ~MarkerSpike() { stop(); }

    bool start() {
        if (running_.exchange(true)) return true;
        if (!openCamera()) {
            closeCamera();
            running_ = false;
            return false;
        }
        worker_ = std::thread([this] { workerLoop(); });
        return true;
    }

    void stop() {
        if (!running_.exchange(false)) return;
        cv_.notify_all();
        closeCamera();  // stops new frames before the worker is joined
        if (worker_.joinable()) worker_.join();
    }

private:
    static constexpr const char *kTag = "PhoneVR_MarkerSpike";
    __attribute__((format(printf, 1, 2))) static void logi(const char *fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        __android_log_vprint(ANDROID_LOG_INFO, kTag, fmt, ap);
        va_end(ap);
    }

    Config cfg_;
    std::atomic<bool> running_{false};
    std::thread worker_;

    // camera objects
    ACameraManager *mgr_ = nullptr;
    ACameraDevice *device_ = nullptr;
    AImageReader *reader_ = nullptr;
    ANativeWindow *window_ = nullptr;
    ACaptureSessionOutputContainer *outputs_ = nullptr;
    ACaptureSessionOutput *output_ = nullptr;
    ACameraOutputTarget *target_ = nullptr;
    ACaptureRequest *request_ = nullptr;
    ACameraCaptureSession *session_ = nullptr;
    int width_ = 0, height_ = 0;
    int timestampSource_ = -1;  // 0 unknown, 1 realtime (CLOCK_BOOTTIME)

    // newest frame handed from the camera callback to the worker
    std::mutex m_;
    std::condition_variable cv_;
    std::vector<uint8_t> frame_;
    bool frameReady_ = false;
    int64_t frameTimestampNs_ = 0;
    int64_t frameArrivedNs_ = 0;

    // counters (camera callback)
    std::atomic<int> captured_{0};
    std::atomic<int> overwritten_{0};

    // ---------------- camera ----------------

    static int64_t nowBoottimeNs() {
        struct timespec ts;
        clock_gettime(CLOCK_BOOTTIME, &ts);
        return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
    }

    static void onDisconnected(void *, ACameraDevice *) { logi("camera disconnected"); }
    static void onError(void *, ACameraDevice *, int error) { logi("camera error %d", error); }
    static void onSessionClosed(void *, ACameraCaptureSession *) {}
    static void onSessionReady(void *, ACameraCaptureSession *) {}
    static void onSessionActive(void *, ACameraCaptureSession *) {}

    static void onImage(void *ctx, AImageReader *reader) {
        static_cast<MarkerSpike *>(ctx)->handleImage(reader);
    }

    void handleImage(AImageReader *reader) {
        AImage *img = nullptr;
        if (AImageReader_acquireLatestImage(reader, &img) != AMEDIA_OK || img == nullptr) return;
        int64_t ts = 0;
        AImage_getTimestamp(img, &ts);
        uint8_t *data = nullptr;
        int len = 0, rowStride = 0;
        if (AImage_getPlaneData(img, 0, &data, &len) == AMEDIA_OK &&
            AImage_getPlaneRowStride(img, 0, &rowStride) == AMEDIA_OK && data != nullptr) {
            captured_++;
            std::lock_guard<std::mutex> lock(m_);
            if (frameReady_) overwritten_++;  // worker hasn't taken the previous one yet
            frame_.resize((size_t) rowStride * height_);
            std::memcpy(frame_.data(), data, std::min((size_t) len, frame_.size()));
            frameStride_ = rowStride;
            frameTimestampNs_ = ts;
            frameArrivedNs_ = nowBoottimeNs();
            frameReady_ = true;
            cv_.notify_one();
        }
        AImage_delete(img);
    }
    int frameStride_ = 0;

    bool openCamera() {
        mgr_ = ACameraManager_create();
        ACameraIdList *ids = nullptr;
        if (ACameraManager_getCameraIdList(mgr_, &ids) != ACAMERA_OK || ids == nullptr) {
            logi("could not list cameras");
            return false;
        }
        std::string chosen;
        ACameraMetadata *meta = nullptr;
        for (int i = 0; i < ids->numCameras && chosen.empty(); i++) {
            ACameraMetadata *m = nullptr;
            if (ACameraManager_getCameraCharacteristics(mgr_, ids->cameraIds[i], &m) != ACAMERA_OK)
                continue;
            ACameraMetadata_const_entry e;
            if (ACameraMetadata_getConstEntry(m, ACAMERA_LENS_FACING, &e) == ACAMERA_OK &&
                e.data.u8[0] == ACAMERA_LENS_FACING_BACK) {
                chosen = ids->cameraIds[i];
                meta = m;
            } else {
                ACameraMetadata_free(m);
            }
        }
        ACameraManager_deleteCameraIdList(ids);
        if (chosen.empty() || meta == nullptr) {
            logi("no back camera found");
            return false;
        }
        describeCamera(chosen.c_str(), meta);
        ACameraMetadata_free(meta);

        ACameraDevice_StateCallbacks devCb = {this, onDisconnected, onError};
        camera_status_t st = ACameraManager_openCamera(mgr_, chosen.c_str(), &devCb, &device_);
        if (st != ACAMERA_OK) {
            logi("openCamera failed (%d). If this is a permission error, grant Camera to PhoneVR "
                 "in the phone's app settings.",
                 (int) st);
            return false;
        }

        if (AImageReader_new(width_, height_, AIMAGE_FORMAT_YUV_420_888, 3, &reader_) != AMEDIA_OK) {
            logi("AImageReader_new failed");
            return false;
        }
        AImageReader_ImageListener listener = {this, onImage};
        AImageReader_setImageListener(reader_, &listener);
        AImageReader_getWindow(reader_, &window_);

        ACaptureSessionOutputContainer_create(&outputs_);
        ACaptureSessionOutput_create(window_, &output_);
        ACaptureSessionOutputContainer_add(outputs_, output_);
        ACameraOutputTarget_create(window_, &target_);
        if (ACameraDevice_createCaptureRequest(device_, TEMPLATE_PREVIEW, &request_) != ACAMERA_OK) {
            logi("createCaptureRequest failed");
            return false;
        }
        ACaptureRequest_addTarget(request_, target_);

        ACameraCaptureSession_stateCallbacks sessCb = {this, onSessionClosed, onSessionReady,
                                                       onSessionActive};
        if (ACameraDevice_createCaptureSession(device_, outputs_, &sessCb, &session_) != ACAMERA_OK) {
            logi("createCaptureSession failed");
            return false;
        }
        if (ACameraCaptureSession_setRepeatingRequest(session_, nullptr, 1, &request_, nullptr) !=
            ACAMERA_OK) {
            logi("setRepeatingRequest failed");
            return false;
        }
        logi("camera %s running, YUV %dx%d", chosen.c_str(), width_, height_);
        return true;
    }

    void closeCamera() {
        if (session_) { ACameraCaptureSession_stopRepeating(session_); ACameraCaptureSession_close(session_); session_ = nullptr; }
        if (request_) { ACaptureRequest_free(request_); request_ = nullptr; }
        if (target_) { ACameraOutputTarget_free(target_); target_ = nullptr; }
        if (output_) { ACaptureSessionOutput_free(output_); output_ = nullptr; }
        if (outputs_) { ACaptureSessionOutputContainer_free(outputs_); outputs_ = nullptr; }
        if (device_) { ACameraDevice_close(device_); device_ = nullptr; }
        if (reader_) { AImageReader_delete(reader_); reader_ = nullptr; window_ = nullptr; }
        if (mgr_) { ACameraManager_delete(mgr_); mgr_ = nullptr; }
    }

    // Logs what the camera can do and picks the stream size.
    void describeCamera(const char *id, ACameraMetadata *meta) {
        ACameraMetadata_const_entry e;

        const char *levels[] = {"LIMITED", "FULL", "LEGACY", "LEVEL_3", "EXTERNAL"};
        int level = -1;
        if (ACameraMetadata_getConstEntry(meta, ACAMERA_INFO_SUPPORTED_HARDWARE_LEVEL, &e) == ACAMERA_OK)
            level = e.data.u8[0];
        if (ACameraMetadata_getConstEntry(meta, ACAMERA_SENSOR_INFO_TIMESTAMP_SOURCE, &e) == ACAMERA_OK)
            timestampSource_ = e.data.u8[0];

        float focalMm = 0, sensorW = 0, sensorH = 0;
        if (ACameraMetadata_getConstEntry(meta, ACAMERA_LENS_INFO_AVAILABLE_FOCAL_LENGTHS, &e) == ACAMERA_OK && e.count > 0)
            focalMm = e.data.f[0];
        if (ACameraMetadata_getConstEntry(meta, ACAMERA_SENSOR_INFO_PHYSICAL_SIZE, &e) == ACAMERA_OK && e.count >= 2) {
            sensorW = e.data.f[0];
            sensorH = e.data.f[1];
        }
        int arrW = 0, arrH = 0;
        if (ACameraMetadata_getConstEntry(meta, ACAMERA_SENSOR_INFO_ACTIVE_ARRAY_SIZE, &e) == ACAMERA_OK && e.count >= 4) {
            arrW = e.data.i32[2];
            arrH = e.data.i32[3];
        }

        // YUV output sizes: pick the 4:3 one closest to the wanted size
        int bestW = 0, bestH = 0;
        long bestScore = -1;
        if (ACameraMetadata_getConstEntry(meta, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &e) == ACAMERA_OK) {
            for (uint32_t i = 0; i + 3 < e.count; i += 4) {
                int fmt = e.data.i32[i], w = e.data.i32[i + 1], h = e.data.i32[i + 2], input = e.data.i32[i + 3];
                if (fmt != AIMAGE_FORMAT_YUV_420_888 || input != 0) continue;
                if (w * 3 != h * 4) continue;
                long score = std::labs((long) w * h - (long) cfg_.wantWidth * cfg_.wantHeight);
                if (bestScore < 0 || score < bestScore) { bestScore = score; bestW = w; bestH = h; }
            }
        }
        if (bestW == 0) { bestW = cfg_.wantWidth; bestH = cfg_.wantHeight; logi("no 4:3 YUV size listed; trying %dx%d anyway", bestW, bestH); }
        width_ = bestW;
        height_ = bestH;

        logi("camera id %s: hardware level %s, image timestamps %s", id,
             (level >= 0 && level < 5) ? levels[level] : "?",
             timestampSource_ == 1 ? "REALTIME (same clock as the sensors)" : "UNKNOWN (not comparable with sensor time)");
        if (focalMm > 0 && sensorW > 0) {
            float hfov = 2.f * std::atan(sensorW / (2.f * focalMm)) * 180.f / (float) M_PI;
            float vfov = 2.f * std::atan(sensorH / (2.f * focalMm)) * 180.f / (float) M_PI;
            logi("lens: focal %.2f mm, sensor %.2f x %.2f mm, active array %dx%d, full-frame FOV %.1f x %.1f deg",
                 focalMm, sensorW, sensorH, arrW, arrH, hfov, vfov);
            logi("approx. focal length at %dx%d: %.0f px (assumes the stream shows the whole sensor width)",
                 bestW, bestH, focalMm / sensorW * bestW);
        } else {
            logi("lens: focal length / sensor size not reported");
        }
    }

    // ---------------- detection + stats ----------------

    struct TagStats {
        int frames = 0;
        double sizeSum = 0, marginSum = 0;
    };

    void workerLoop() {
        MarkerDetector detector(cfg_.detector);
        std::vector<uint8_t> local;
        std::vector<MarkerDetection> dets;
        std::map<int, TagStats> tags;
        int processed = 0;
        double detectSumMs = 0, detectMaxMs = 0, latencySumMs = 0;
        int latencyCount = 0;
        int lastCaptured = 0, lastOverwritten = 0;
        auto windowStart = std::chrono::steady_clock::now();

        while (running_) {
            int stride = 0;
            int64_t ts = 0, arrived = 0;
            {
                std::unique_lock<std::mutex> lock(m_);
                cv_.wait_for(lock, std::chrono::milliseconds(200), [this] { return frameReady_ || !running_; });
                if (!running_) break;
                if (!frameReady_) continue;
                local.swap(frame_);
                stride = frameStride_;
                ts = frameTimestampNs_;
                arrived = frameArrivedNs_;
                frameReady_ = false;
            }
            auto t0 = std::chrono::steady_clock::now();
            detector.detect(local.data(), width_, height_, stride, dets);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            processed++;
            detectSumMs += ms;
            detectMaxMs = std::max(detectMaxMs, ms);
            if (timestampSource_ == 1) {
                latencySumMs += (arrived - ts) * 1e-6;
                latencyCount++;
            }
            for (const auto &d : dets) {
                TagStats &s = tags[d.id];
                s.frames++;
                s.sizeSum += d.sizePx;
                s.marginSum += d.margin;
            }

            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - windowStart).count();
            if (elapsed >= 1.0) {
                int cap = captured_.load(), over = overwritten_.load();
                logi("%.1f fps from camera, %.1f processed (%d frames skipped) | detect %.1f ms avg, %.1f ms max",
                     (cap - lastCaptured) / elapsed, processed / elapsed, over - lastOverwritten,
                     processed ? detectSumMs / processed : 0.0, detectMaxMs);
                if (latencyCount) logi("frame age on arrival: %.1f ms avg (camera exposure + readout + delivery)", latencySumMs / latencyCount);
                if (tags.empty()) {
                    logi("no tags seen in the last second");
                } else {
                    for (const auto &kv : tags)
                        logi("  tag %d: seen in %d/%d frames, mean size %.0f px, margin %.0f", kv.first,
                             kv.second.frames, processed, kv.second.sizeSum / kv.second.frames,
                             kv.second.marginSum / kv.second.frames);
                }
                lastCaptured = cap;
                lastOverwritten = over;
                processed = 0;
                detectSumMs = detectMaxMs = latencySumMs = 0;
                latencyCount = 0;
                tags.clear();
                windowStart = std::chrono::steady_clock::now();
            }
        }
    }
};

}  // namespace phonevr

#endif  // PHONEVR_MARKER_SPIKE_H
