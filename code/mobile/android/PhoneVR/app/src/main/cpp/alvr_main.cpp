#include "alvr_client_core.h"
#include "arcore_c_api.h"
#include "cardboard.h"
#include <GLES3/gl3.h>
#include <EGL/egl.h>
#include <algorithm>
#include <android/log.h>
#include <cstring>
#include <deque>
#include <jni.h>
#include <map>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

#include "utils.h"


#ifndef LOG
#define LOG(...) __android_log_print(ANDROID_LOG_INFO, "ALVR_EGLCHK", __VA_ARGS__)
#endif
#define EGLCHK(where) LOG("EGLCHK " where " ctx=%p surf=%p", eglGetCurrentContext(), eglGetCurrentSurface(EGL_DRAW))

struct EglStateGuard {
    EGLDisplay dpy = eglGetCurrentDisplay();
    EGLContext ctx = eglGetCurrentContext();
    EGLSurface draw = eglGetCurrentSurface(EGL_DRAW);
    EGLSurface read = eglGetCurrentSurface(EGL_READ);
    ~EglStateGuard() {
        if (ctx != EGL_NO_CONTEXT && eglGetCurrentContext() != ctx)
            eglMakeCurrent(dpy, draw, read, ctx);
    }
};

// TEMP DEBUG: burn texture names before creating the lobby swapchain, so the names handed to the
// ALVR library (17,18 instead of 1,2) can't coincide with names its own EGL context already uses.
// Remove once the context-sharing question is settled.
#define PVR_TEXTURE_NAME_TEST 1

// The ALVR graphics code aborts the process when it sees a GL error. glGetError() is sticky, so
// an error left behind by app/Cardboard code would be blamed on the library call. Drain (and
// log) any pending errors before each library call.
static void drainGlErrors(const char *where) {
    GLenum e;
    int n = 0;
    while ((e = glGetError()) != GL_NO_ERROR && n++ < 16)
        LOG("Pending GL error 0x%x before %s", e, where);
}

// One-time GL driver info, to check feature support (external images) when debugging.
static void logGlInfo() {
    const char *ver = (const char *) glGetString(GL_VERSION);
    const char *ren = (const char *) glGetString(GL_RENDERER);
    const char *ext = (const char *) glGetString(GL_EXTENSIONS);
    LOG("GLINFO version=%s renderer=%s", ver ? ver : "?", ren ? ren : "?");
    LOG("GLINFO image_external_essl3=%d image_external=%d",
        ext && strstr(ext, "GL_OES_EGL_image_external_essl3") != nullptr,
        ext && strstr(ext, "GL_OES_EGL_image_external") != nullptr);
}

// The header says later DecoderConfig events are ignored until reconnection.
static bool g_decoderCreated = false;

// alvr_*_opengl() calls in this ALVR version can leave the thread with no current EGL context.
// Run them under a guard so the app's context/surface is rebound right after each call.
#define ALVR_GL_CALL(call)                \
    do {                                  \
        drainGlErrors(#call);             \
        EglStateGuard _eglGuard;          \
        call;                             \
    } while (0)


uint64_t HEAD_ID = alvr_path_string_to_id("/user/head");

// v21 removed the alvr_get_head_prediction_offset_ns() C export (it's now
// ClientCoreContext::get_total_prediction_offset(), a Rust-only method used internally by
// client_openxr). There is currently no C-ABI replacement, so this is a fixed heuristic in the
// meantime.
// TODO(upstream): propose re-exposing get_total_prediction_offset() via c_api.rs so native
// clients don't have to guess this.
const uint64_t ESTIMATED_PREDICTION_OFFSET_NS = 30e6;

// Note: the Cardboard SDK cannot estimate display time and an heuristic is used instead.
const uint64_t VSYNC_QUEUE_INTERVAL_NS = 50e6;
const float FLOOR_HEIGHT = 1.5;
const int MAXIMUM_TRACKING_FRAMES = 360;

struct NativeContext {
    JavaVM *javaVm = nullptr;
    jobject javaContext = nullptr;

    CardboardHeadTracker *headTracker = nullptr;
    CardboardLensDistortion *lensDistortion = nullptr;
    CardboardDistortionRenderer *distortionRenderer = nullptr;

    // ARCore (positional tracking). ARCore calls that touch the session must not run
    // concurrently, so every use of arSession is guarded by arMutex. ArSession_update() must
    // additionally run on the render thread, with the app's EGL context current.
    bool arcoreEnabled = false;
    ArSession *arSession = nullptr;
    ArFrame *arFrame = nullptr;
    GLuint arTexture = 0;
    std::mutex arMutex;

    int screenWidth = 0;
    int screenHeight = 0;

    bool renderingParamsChanged = true;
    bool glContextRecreated = false;

    bool running = false;
    bool streaming = false;
    std::thread inputThread;

    // Use one texture per eye, no need for swapchains.
    GLuint lobbyTextures[2] = {0, 0};
    GLuint streamTextures[2] = {0, 0};

    float eyeOffsets[2] = {0.0, 0.0};
    AlvrFov fovArr[2] = {};
    AlvrDeviceMotion deviceMotion = {};

    // Cache of the most recently matched frame metadata, since alvr_report_compositor_start()
    // can legitimately return false (no match for this exact frame), in which case the caller
    // is expected to reuse the previous metadata.
    bool hasFrameMetadata = false;
    AlvrVideoFrameMetadata lastFrameMetadata = {};

    NativeContext() {
        memset(&fovArr, 0, sizeof(fovArr));
        memset(&deviceMotion, 0, sizeof(deviceMotion));
    }
};

NativeContext CTX = {};

int64_t GetBootTimeNano() {
    struct timespec res = {};
    clock_gettime(CLOCK_BOOTTIME, &res);
    return (res.tv_sec * 1e9) + res.tv_nsec;
}

// Inverse unit quaternion
AlvrQuat inverseQuat(AlvrQuat q) { return {-q.x, -q.y, -q.z, q.w}; }

void cross(float a[3], float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

void quatVecMultiply(AlvrQuat q, float v[3], float out[3]) {
    float rv[3], rrv[3];
    float r[3] = {q.x, q.y, q.z};
    cross(r, v, rv);
    cross(r, rv, rrv);
    for (int i = 0; i < 3; i++) {
        out[i] = v[i] + 2 * (q.w * rv[i] + rrv[i]);
    }
}

void offsetPosWithQuat(AlvrQuat q, float offset[3], float outPos[3]) {
    float rotatedOffset[3];
    quatVecMultiply(q, offset, rotatedOffset);

    outPos[0] -= rotatedOffset[0];
    outPos[1] -= rotatedOffset[1] - FLOOR_HEIGHT;
    outPos[2] -= rotatedOffset[2];
}

AlvrFov getFov(CardboardEye eye) {
    float f[4];
    CardboardLensDistortion_getFieldOfView(CTX.lensDistortion, eye, f);

    AlvrFov fov = {};
    fov.left = -f[0];
    fov.right = f[1];
    fov.up = f[3];
    fov.down = -f[2];

    return fov;
}

// Head pose, in world/tracking space (not head-relative). Rotation currently comes straight from
// the Cardboard SDK's gyro-only sensor fusion, so it has no absolute position and will drift over
// time - this is the exact function that the planned ARCore + complementary filter work (a
// separate, later step) will extend to fuse in ARCore's 6DoF pose.
AlvrPose getPose(uint64_t timestampNs) {
    AlvrPose pose = {};

    float pos[3];
    float q[4];
    CardboardHeadTracker_getPose(CTX.headTracker, (int64_t) timestampNs, kLandscapeLeft, pos, q);

    auto inverseOrientation = AlvrQuat{q[0], q[1], q[2], q[3]};
    pose.orientation = inverseQuat(inverseOrientation);

    // Cardboard-only tracking has no position signal (3DoF), so position stays at the origin
    // for now.

    return pose;
}

// STEP 2a (diagnostic only): update ARCore once per render frame and log what it reports about
// once per second. The result is NOT used for tracking yet; this only checks that the session
// runs on the render thread with the app's EGL context current. Must be called from the render
// thread.
static void pollARCoreDiagnostics() {
    if (!CTX.arcoreEnabled || CTX.arTexture == 0) {
        return;
    }
    if (eglGetCurrentContext() == EGL_NO_CONTEXT) {
        info("ARCore ERROR: pollARCoreDiagnostics called without an EGL context");
        return;
    }

    std::lock_guard<std::mutex> lock(CTX.arMutex);
    if (CTX.arSession == nullptr || CTX.arFrame == nullptr) {
        return;
    }

    static uint64_t lastLogNs = 0;
    uint64_t nowNs = GetBootTimeNano();
    bool shouldLog = nowNs - lastLogNs >= 1000000000ULL;

    ArStatus status = ArSession_update(CTX.arSession, CTX.arFrame);
    if (status != AR_SUCCESS) {
        if (shouldLog) {
            lastLogNs = nowNs;
            info("ARCore ERROR: ArSession_update failed (status %d)", (int) status);
        }
        return;
    }

    ArCamera *arCamera = nullptr;
    ArFrame_acquireCamera(CTX.arSession, CTX.arFrame, &arCamera);

    ArTrackingState trackingState;
    ArCamera_getTrackingState(CTX.arSession, arCamera, &trackingState);

    if (shouldLog) {
        lastLogNs = nowNs;
        if (trackingState == AR_TRACKING_STATE_TRACKING) {
            ArPose *arPose = nullptr;
            ArPose_create(CTX.arSession, nullptr, &arPose);
            ArCamera_getDisplayOrientedPose(CTX.arSession, arCamera, arPose);
            // {qx, qy, qz, qw, tx, ty, tz}
            float raw[7] = {};
            ArPose_getPoseRaw(CTX.arSession, arPose, raw);
            ArPose_destroy(arPose);
            info("ARCore: tracking, position=(%.3f, %.3f, %.3f) quat=(%.3f, %.3f, %.3f, %.3f)",
                 raw[4], raw[5], raw[6], raw[0], raw[1], raw[2], raw[3]);
        } else {
            info("ARCore: not tracking (state %d)", (int) trackingState);
        }
    }

    ArCamera_release(arCamera);
}

// Local-space (head-at-origin) per-eye view parameters: just the eye offset and FOV. These rarely
// change (only when the lens distortion / viewer profile is recalculated), so this is sent once
// via alvr_send_view_params() rather than on every tracking update - unlike the v20 API, the
// moving head pose is no longer bundled into this call.
void sendViewParams() {
    AlvrViewParams viewParams[2] = {};

    for (int eye = 0; eye < 2; eye++) {
        viewParams[eye].pose.orientation = AlvrQuat{0.0, 0.0, 0.0, 1.0};
        viewParams[eye].pose.position[0] = CTX.eyeOffsets[eye];
        viewParams[eye].pose.position[1] = 0.0;
        viewParams[eye].pose.position[2] = 0.0;
        viewParams[eye].fov = CTX.fovArr[eye];
    }

    alvr_send_view_params(viewParams);
}

void inputThread() {
    auto deadline = std::chrono::steady_clock::now();

    info("inputThread: thread staring...");
    while (CTX.streaming) {
        auto targetTimestampNs = GetBootTimeNano() + ESTIMATED_PREDICTION_OFFSET_NS;

        CTX.deviceMotion.device_id = HEAD_ID;
        CTX.deviceMotion.pose = getPose(targetTimestampNs);

        alvr_send_tracking(targetTimestampNs, &CTX.deviceMotion, 1, nullptr, nullptr);

        deadline += std::chrono::nanoseconds((uint64_t) (1e9 / 60.f / 3));
        std::this_thread::sleep_until(deadline);
    }
}

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
    CTX.javaVm = vm;
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_initializeNative(
    JNIEnv *env, jobject obj, jint screenWidth, jint screenHeight, jfloat refreshRate,jboolean enableARCore) {
    CTX.javaContext = env->NewGlobalRef(obj);
    info("initializeNative: enableARCore=%d", (int) enableARCore);

    uint32_t viewWidth = (std::max(screenWidth, screenHeight) / 2) & ~15u;  // multiple of 16
    uint32_t viewHeight = std::min(screenWidth, screenHeight) & ~7u;        // multiple of 8

    // Route the Rust library's log output (including the reason for its aborts) to logcat.
    alvr_initialize_logging();

    alvr_initialize_android_context((void *) CTX.javaVm, (void *) CTX.javaContext);

    static float refreshRatesBuffer[1];
    refreshRatesBuffer[0] = refreshRate;

    AlvrClientCapabilities caps = {};
    caps.default_view_width = viewWidth;
    caps.default_view_height = viewHeight;
    caps.max_view_width = viewWidth;
    caps.max_view_height = viewHeight;
    caps.refresh_rates = refreshRatesBuffer;
    caps.refresh_rates_count = 1;
    caps.foveated_encoding = false;
    caps.encoder_high_profile = true;
    caps.encoder_10_bits = false;
    caps.encoder_av1 = false;
    caps.prefer_10bit = false;
    caps.preferred_encoding_gamma = 1.0;
    caps.prefer_hdr = false;

    alvr_initialize(caps);

    Cardboard_initializeAndroid(CTX.javaVm, CTX.javaContext);
    CTX.headTracker = CardboardHeadTracker_create();

    CTX.arcoreEnabled = (bool) enableARCore;
    if (CTX.arcoreEnabled) {
        if (ArSession_create(env, CTX.javaContext, &CTX.arSession) != AR_SUCCESS) {
            info("ARCore ERROR: could not create session (is Google Play Services for AR "
                 "installed?); continuing without ARCore");
            CTX.arSession = nullptr;
            CTX.arcoreEnabled = false;
        }
    }

    if (CTX.arcoreEnabled) {
        ArConfig *arConfig = nullptr;
        ArConfig_create(CTX.arSession, &arConfig);

        // Explicitly disable everything we don't need, to preserve CPU and battery.
        ArConfig_setDepthMode(CTX.arSession, arConfig, AR_DEPTH_MODE_DISABLED);
        ArConfig_setLightEstimationMode(
            CTX.arSession, arConfig, AR_LIGHT_ESTIMATION_MODE_DISABLED);
        ArConfig_setPlaneFindingMode(
            CTX.arSession, arConfig, AR_PLANE_FINDING_MODE_HORIZONTAL_AND_VERTICAL);
        ArConfig_setCloudAnchorMode(CTX.arSession, arConfig, AR_CLOUD_ANCHOR_MODE_DISABLED);
        // Don't block waiting for a new camera image.
        ArConfig_setUpdateMode(CTX.arSession, arConfig, AR_UPDATE_MODE_LATEST_CAMERA_IMAGE);

        ArStatus configureStatus = ArSession_configure(CTX.arSession, arConfig);
        ArConfig_destroy(arConfig);

        if (configureStatus != AR_SUCCESS) {
            info("ARCore ERROR: could not configure session (status %d); continuing without "
                 "ARCore",
                 (int) configureStatus);
            ArSession_destroy(CTX.arSession);
            CTX.arSession = nullptr;
            CTX.arcoreEnabled = false;
        } else {
            ArFrame_create(CTX.arSession, &CTX.arFrame);
            info("ARCore session created and configured");
        }
    }
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_destroyNative(JNIEnv *,
                                                                                        jobject) {
    alvr_destroy_opengl();
    alvr_destroy();

    CardboardHeadTracker_destroy(CTX.headTracker);
    CTX.headTracker = nullptr;
    CardboardLensDistortion_destroy(CTX.lensDistortion);
    CTX.lensDistortion = nullptr;
    CardboardDistortionRenderer_destroy(CTX.distortionRenderer);
    CTX.distortionRenderer = nullptr;

    {
        std::lock_guard<std::mutex> lock(CTX.arMutex);
        if (CTX.arFrame != nullptr) {
            ArFrame_destroy(CTX.arFrame);
            CTX.arFrame = nullptr;
        }
        if (CTX.arSession != nullptr) {
            ArSession_destroy(CTX.arSession);
            CTX.arSession = nullptr;
        }
        CTX.arcoreEnabled = false;
    }
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_resumeNative(JNIEnv *,
                                                                                       jobject) {
    CardboardHeadTracker_resume(CTX.headTracker);

    {
        std::lock_guard<std::mutex> lock(CTX.arMutex);
        if (CTX.arcoreEnabled && CTX.arSession != nullptr) {
            ArStatus arStatus = ArSession_resume(CTX.arSession);
            if (arStatus != AR_SUCCESS) {
                info("ARCore ERROR: failed to resume session (status %d)", (int) arStatus);
            }
        }
    }

    CTX.renderingParamsChanged = true;

    uint8_t *buffer;
    int size;
    CardboardQrCode_getSavedDeviceParams(&buffer, &size);
    if (size == 0) {
        CardboardQrCode_scanQrCodeAndSaveDeviceParams();
    }
    CardboardQrCode_destroy(buffer);

    CTX.running = true;
    alvr_resume();
    
    // at the start of renderNative, right after alvr_resume(...), and right before the final draw:
    LOG("EGLCHK ctx=%p surf=%p", eglGetCurrentContext(), eglGetCurrentSurface(EGL_DRAW));
    EGLCHK("after alvr_resume()");
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_pauseNative(JNIEnv *,
                                                                                      jobject) {
    alvr_pause();

    if (CTX.running) {
        CTX.running = false;
    }

    CardboardHeadTracker_pause(CTX.headTracker);

    // Releases the camera. (The v20 code never paused the ARCore session.)
    {
        std::lock_guard<std::mutex> lock(CTX.arMutex);
        if (CTX.arcoreEnabled && CTX.arSession != nullptr) {
            ArSession_pause(CTX.arSession);
        }
    }
}

extern "C" JNIEXPORT void JNICALL
Java_viritualisres_phonevr_ALVRActivity_surfaceCreatedNative(JNIEnv *, jobject) {
    logGlInfo();
    ALVR_GL_CALL(alvr_initialize_opengl());

    CTX.glContextRecreated = true;
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_setScreenResolutionNative(
    JNIEnv *, jobject, jint width, jint height) {
    CTX.screenWidth = width;
    CTX.screenHeight = height;

    CTX.renderingParamsChanged = true;
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_sendBatteryLevel(
    JNIEnv *, jobject, jfloat level, jboolean plugged) {
    alvr_send_battery(HEAD_ID, level, plugged);
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_renderNative(JNIEnv *,
                                                                                       jobject) {
    // at the start of renderNative, right after alvr_resume(...), and right before the final draw:
    LOG("EGLCHK ctx=%p surf=%p", eglGetCurrentContext(), eglGetCurrentSurface(EGL_DRAW));
    EGLCHK("Start of renderNative()");

    if (CTX.renderingParamsChanged) {
        info("renderingParamsChanged, processing new params");
        uint8_t *buffer;
        int size;
        CardboardQrCode_getSavedDeviceParams(&buffer, &size);

        if (size == 0) {
            return;
        }

        info("renderingParamsChanged, sending new params to alvr");
        if (CTX.lensDistortion) {
            CardboardLensDistortion_destroy(CTX.lensDistortion);
            CTX.lensDistortion = nullptr;
        }
        info("renderingParamsChanged, destroyed distortion");
        CTX.lensDistortion =
            CardboardLensDistortion_create(buffer, size, CTX.screenWidth, CTX.screenHeight);

        CardboardQrCode_destroy(buffer);
        *buffer = 0;

        if (CTX.distortionRenderer) {
            CardboardDistortionRenderer_destroy(CTX.distortionRenderer);
            CTX.distortionRenderer = nullptr;
        }
        const CardboardOpenGlEsDistortionRendererConfig config{kGlTexture2D};
        CTX.distortionRenderer = CardboardOpenGlEs2DistortionRenderer_create(&config);

        for (int eye = 0; eye < 2; eye++) {
            CardboardMesh mesh;
            CardboardLensDistortion_getDistortionMesh(
                CTX.lensDistortion, (CardboardEye) eye, &mesh);
            CardboardDistortionRenderer_setMesh(
                CTX.distortionRenderer, &mesh, (CardboardEye) eye);

            float matrix[16] = {};
            CardboardLensDistortion_getEyeFromHeadMatrix(
                CTX.lensDistortion, (CardboardEye) eye, matrix);
            CTX.eyeOffsets[eye] = matrix[12];
        }

        CTX.fovArr[kLeft] = getFov(kLeft);
        CTX.fovArr[kRight] = getFov(kRight);

        info("renderingParamsChanged, sending new view configs (FOV) to alvr");
        sendViewParams();
    }

    // Note: if GL context is recreated, old resources are already freed.
    if (CTX.renderingParamsChanged && !CTX.glContextRecreated) {
        info("Pausing ALVR since glContext is not recreated, deleting textures");
        ALVR_GL_CALL(alvr_pause_opengl());

        GL(glDeleteTextures(2, CTX.lobbyTextures));
    }

    if (CTX.renderingParamsChanged || CTX.glContextRecreated) {
        info("Rebuilding, binding textures, Resuming ALVR since glContextRecreated %b, "
             "renderingParamsChanged %b",
             CTX.renderingParamsChanged,
             CTX.glContextRecreated);
#ifdef PVR_TEXTURE_NAME_TEST
        {
            static bool burned = false;
            if (!burned) {
                GLuint burn[16];
                glGenTextures(16, burn);
                burned = true;
            }
        }
#endif
        GL(glGenTextures(2, CTX.lobbyTextures));

        for (auto &lobbyTexture : CTX.lobbyTextures) {
            GL(glBindTexture(GL_TEXTURE_2D, lobbyTexture));
            GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
            GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
            GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            GL(glTexImage2D(GL_TEXTURE_2D,
                            0,
                            GL_RGB,
                            CTX.screenWidth / 2,
                            CTX.screenHeight,
                            0,
                            GL_RGB,
                            GL_UNSIGNED_BYTE,
                            nullptr));
        }

        // The ALVR library renders in its own EGL context (now a share context of this one).
        // Object creation/changes in this context are only guaranteed visible to the other
        // context after a flush/finish, so finish before handing the texture names over.
        glFinish();
        LOG("LOBBYTEX names=%u,%u isTexture=%d,%d size=%dx%d",
            (unsigned) CTX.lobbyTextures[0],
            (unsigned) CTX.lobbyTextures[1],
            (int) glIsTexture(CTX.lobbyTextures[0]),
            (int) glIsTexture(CTX.lobbyTextures[1]),
            CTX.screenWidth / 2,
            CTX.screenHeight);

        const uint32_t *targetViews[2] = {(const uint32_t *) &CTX.lobbyTextures[0],
                                          (const uint32_t *) &CTX.lobbyTextures[1]};
        // v21 dropped the trailing bool (was "always direct") - resume takes 4 args now.
        // alvr_resume_opengl() leaves the thread with no current EGL context (observed:
        // ctx/surf go to 0x0), which makes every later GL call fail and GLSurfaceView stop
        // rendering. The guard captures the app's EGL state before the call and rebinds it
        // immediately afterwards.
        EGLCHK("before alvr_resume_opengl");
        {
            EglStateGuard guard;
            alvr_resume_opengl(CTX.screenWidth / 2, CTX.screenHeight, targetViews, 1);
        }
        EGLCHK("after alvr_resume_opengl (restored)");

        // ARCore needs a GL texture to write camera frames into. We never draw the camera
        // image, but ARCore requires one. If the GL context was recreated the old texture is
        // already gone; otherwise delete it so it doesn't leak.
        if (CTX.arcoreEnabled && CTX.arSession != nullptr) {
            if (CTX.arTexture != 0 && !CTX.glContextRecreated) {
                glDeleteTextures(1, &CTX.arTexture);
            }
            GLuint arTextureIds[1] = {0};
            glGenTextures(1, arTextureIds);
            GL(glBindTexture(GL_TEXTURE_2D, arTextureIds[0]));
            GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
            GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
            GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

            {
                std::lock_guard<std::mutex> lock(CTX.arMutex);
                CTX.arTexture = arTextureIds[0];
                ArSession_setCameraTextureName(CTX.arSession, CTX.arTexture);
            }
            info("ARCore camera texture initialized: %u", (unsigned) CTX.arTexture);
        }

        CTX.renderingParamsChanged = false;
        CTX.glContextRecreated = false;
    }

    AlvrEvent event;
    while (alvr_poll_event(&event)) {
        if (event.tag == ALVR_EVENT_HUD_MESSAGE_UPDATED) {
            auto message_length = alvr_hud_message(nullptr);
            auto message_buffer = std::vector<char>(message_length);

            alvr_hud_message(&message_buffer[0]);
            info("ALVR Poll Event: HUD Message Update - %s", &message_buffer[0]);

            // 3) inside the HUD branch, right after the call
            if (message_length > 0)
                ALVR_GL_CALL(alvr_update_hud_message_opengl(&message_buffer[0]));
            EGLCHK("after update_hud_message");
        }
        if (event.tag == ALVR_EVENT_STREAMING_STARTED) {
            info("ALVR Poll Event: ALVR_EVENT_STREAMING_STARTED, generating and binding "
                 "textures...");
            auto &started = event.STREAMING_STARTED;

            info("Streaming started: %dx%d, refresh hint=%f, gamma=%f, hdr=%b, ffe=%b",
                 started.view_width,
                 started.view_height,
                 started.refresh_rate_hint,
                 started.encoding_gamma,
                 started.enable_hdr,
                 started.enable_foveated_encoding);

            // v21 delivers the negotiated foveation params directly in the event - no more
            // manual settings-JSON parsing needed here (settings.video.foveated_encoding, as
            // the v20 code did).
            CTX.hasFrameMetadata = false;

            GL(glGenTextures(2, CTX.streamTextures));

            for (auto &streamTexture : CTX.streamTextures) {
                GL(glBindTexture(GL_TEXTURE_2D, streamTexture));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
                GL(glTexImage2D(GL_TEXTURE_2D,
                                0,
                                GL_RGB,
                                started.view_width,
                                started.view_height,
                                0,
                                GL_RGB,
                                GL_UNSIGNED_BYTE,
                                nullptr));
            }

            CTX.fovArr[0] = getFov((CardboardEye) 0);
            CTX.fovArr[1] = getFov((CardboardEye) 1);
            sendViewParams();

            info("ALVR Poll Event: ALVR_EVENT_STREAMING_STARTED, View configs updated...");

            auto leftIntHandle = (uint32_t) CTX.streamTextures[0];
            auto rightIntHandle = (uint32_t) CTX.streamTextures[1];
            const uint32_t *textureHandles[2] = {&leftIntHandle, &rightIntHandle};

            auto render_config = AlvrStreamConfig{};
            render_config.view_resolution_width = started.view_width;
            render_config.view_resolution_height = started.view_height;
            render_config.swapchain_textures = textureHandles;
            render_config.swapchain_length = 1;

            render_config.enable_foveation = started.enable_foveated_encoding;
            render_config.foveated_encoding = started.foveated_encoding;

            // Not opting into server-side upscaling yet.
            render_config.enable_upscaling = false;

            ALVR_GL_CALL(alvr_start_stream_opengl(render_config));

            info("ALVR Poll Event: ALVR_EVENT_STREAMING_STARTED, opengl stream started and "
                 "input Thread started...");
            CTX.streaming = true;
            CTX.inputThread = std::thread(inputThread);
        } else if (event.tag == ALVR_EVENT_DECODER_CONFIG) {

            /*
             * v21 no longer sends the decoder config in the ALVR_EVENT_STREAMING_STARTED event,
             * so we have to create the decoder here instead. The decoder config is sent in a
             * separate event, ALVR_EVENT_DECODER_CONFIG, which is guaranteed to arrive before
             * the first frame.
             */
            /*
            if (!g_decoderCreated) {
                auto configSize = alvr_get_decoder_config(nullptr);
                auto configBuffer = std::vector<char>(configSize);
                alvr_get_decoder_config(configBuffer.data());

                AlvrDecoderConfig decoderConfig = {};
                decoderConfig.codec = event.DECODER_CONFIG.codec;
                decoderConfig.force_software_decoder = true;
                decoderConfig.max_buffering_frames = 1.5f;
                decoderConfig.buffering_history_weight = 0.9f;
                decoderConfig.options = nullptr;
                decoderConfig.options_count = 0;
                decoderConfig.config_buffer =
                    reinterpret_cast<const uint8_t *>(configBuffer.data());
                decoderConfig.config_buffer_size = configSize;

                info("ALVR Poll Event: ALVR_EVENT_DECODER_CONFIG, codec=%d, config bytes=%llu",
                     (int) decoderConfig.codec, (unsigned long long) configSize);
                alvr_create_decoder(decoderConfig);
                g_decoderCreated = true;
            }
            */

            if (!g_decoderCreated) {
                auto configSize = alvr_get_decoder_config(nullptr);
                auto configBuffer = std::vector<char>(configSize);
                alvr_get_decoder_config(configBuffer.data());

                // false = try the phone's hardware decoder; flip to true if it fails again.
                constexpr bool FORCE_SOFTWARE_DECODER = false;

                // Same options the stock ALVR client gets from the server's defaults.
                AlvrMediacodecOption decoderOptions[3] = {};
                decoderOptions[0].key = "operating-rate";
                decoderOptions[0].ty = ALVR_MEDIACODEC_PROP_TYPE_INT32;
                decoderOptions[0].value.int32 = INT32_MAX;
                decoderOptions[1].key = "priority";
                decoderOptions[1].ty = ALVR_MEDIACODEC_PROP_TYPE_INT32;
                decoderOptions[1].value.int32 = 0;
                decoderOptions[2].key = "vendor.qti-ext-dec-low-latency.enable";
                decoderOptions[2].ty = ALVR_MEDIACODEC_PROP_TYPE_INT32;
                decoderOptions[2].value.int32 = 1;

                AlvrDecoderConfig decoderConfig = {};
                decoderConfig.codec = event.DECODER_CONFIG.codec;
                decoderConfig.force_software_decoder = FORCE_SOFTWARE_DECODER;
                decoderConfig.max_buffering_frames = 1.5f;
                decoderConfig.buffering_history_weight = 0.9f;
                decoderConfig.options = decoderOptions;
                decoderConfig.options_count = 3;
                decoderConfig.config_buffer =
                    reinterpret_cast<const uint8_t *>(configBuffer.data());
                decoderConfig.config_buffer_size = configSize;

                info("ALVR Poll Event: ALVR_EVENT_DECODER_CONFIG, codec=%d, config bytes=%llu, "
                     "software=%d",
                     (int) decoderConfig.codec, (unsigned long long) configSize,
                     (int) FORCE_SOFTWARE_DECODER);
                alvr_create_decoder(decoderConfig);
                g_decoderCreated = true;
            }

        } else if (event.tag == ALVR_EVENT_STREAMING_STOPPED) {
            info("ALVR Poll Event: ALVR_EVENT_STREAMING_STOPPED, Waiting for inputThread to "
                 "join...");
            CTX.streaming = false;
            CTX.inputThread.join();

            if (g_decoderCreated) {
                alvr_destroy_decoder();
                g_decoderCreated = false;
            }

            GL(glDeleteTextures(2, CTX.streamTextures));
            info("ALVR Poll Event: ALVR_EVENT_STREAMING_STOPPED, Stream stopped deleted "
                 "textures.");
        }
    }
    // 4) right after the closing brace of the `while (alvr_poll_event(&event))` loop,
    //    before `CardboardEyeTextureDescription viewsDescs[2] = {};`
    EGLCHK("after event loop");

    // Step 2a: diagnostics only; the ARCore result is not used for tracking yet.
    pollARCoreDiagnostics();

    CardboardEyeTextureDescription viewsDescs[2] = {};
    for (auto &viewsDesc : viewsDescs) {
        viewsDesc.left_u = 0.0;
        viewsDesc.right_u = 1.0;
        viewsDesc.top_v = 1.0;
        viewsDesc.bottom_v = 0.0;
    }

    if (CTX.streaming) {
        void *streamHardwareBuffer = nullptr;
        uint64_t timestampNs = 0;

        if (!alvr_get_frame(&timestampNs, &streamHardwareBuffer)) {
            return;
        }

        // v21 no longer returns view params from alvr_get_frame() directly; fetch the
        // server-aligned metadata (view params + foveation centers) matched to this frame here
        // instead. If there's no match for this exact frame, fall back to the last known-good
        // metadata, per the API doc for this function.
        AlvrVideoFrameMetadata metadata;
        if (alvr_report_compositor_start(timestampNs, &metadata)) {
            CTX.lastFrameMetadata = metadata;
            CTX.hasFrameMetadata = true;
        } else if (CTX.hasFrameMetadata) {
            metadata = CTX.lastFrameMetadata;
        } else {
            // No metadata at all yet (first frame) - nothing sensible to reproject against.
            return;
        }

        // Current best head-orientation estimate, for reprojecting away the latency between
        // when the server rendered this frame and now. The Cardboard SDK's own distortion
        // renderer has no reprojection support of its own; this reprojection_rotation is what
        // lets v21 do it properly even on this simple client.
        AlvrPose currentPose = getPose(GetBootTimeNano() + VSYNC_QUEUE_INTERVAL_NS);

        AlvrStreamViewParams streamViewParams[2] = {};
        for (int eye = 0; eye < 2; eye++) {
            streamViewParams[eye].swapchain_index = 0;
            streamViewParams[eye].reprojection_rotation =
                alvr_rotation_delta(metadata.view_params[eye].pose.orientation,
                                     currentPose.orientation);
            streamViewParams[eye].fov = metadata.view_params[eye].fov;
        }

        const float(*foveationCenters)[2][2] =
            metadata.has_foveation_centers ? &metadata.foveation_center_shifts : nullptr;

        EGLCHK("before alvr_render_stream_opengl()");
        ALVR_GL_CALL(
            alvr_render_stream_opengl(streamHardwareBuffer, streamViewParams, foveationCenters));
        EGLCHK("after alvr_render_stream_opengl()");

        alvr_report_submit(timestampNs, 0);

        viewsDescs[0].texture = CTX.streamTextures[0];
        viewsDescs[1].texture = CTX.streamTextures[1];
    } else {
        AlvrPose pose = getPose(GetBootTimeNano() + VSYNC_QUEUE_INTERVAL_NS);

        AlvrLobbyViewParams viewInputs[2] = {};
        for (int eye = 0; eye < 2; eye++) {
            float headToEye[3] = {CTX.eyeOffsets[eye], 0.0, 0.0};
            // offset head pos to Eye Position
            offsetPosWithQuat(
                pose.orientation, headToEye, viewInputs[eye].view_params.pose.position);

            viewInputs[eye].view_params.pose.orientation = pose.orientation;
            viewInputs[eye].view_params.fov = getFov((CardboardEye) eye);
            viewInputs[eye].swapchain_index = 0;
        }
        static int lobbyTexLogs = 0;
        if (lobbyTexLogs++ < 3) {
            LOG("LOBBYTEX before lobby render: names=%u,%u isTexture=%d,%d",
                (unsigned) CTX.lobbyTextures[0],
                (unsigned) CTX.lobbyTextures[1],
                (int) glIsTexture(CTX.lobbyTextures[0]),
                (int) glIsTexture(CTX.lobbyTextures[1]));
        }
        EGLCHK("before alvr_render_lobby_opengl()");
        ALVR_GL_CALL(alvr_render_lobby_opengl(viewInputs, /*render_background=*/true));
        EGLCHK("after alvr_render_lobby_opengl()");

        viewsDescs[0].texture = CTX.lobbyTextures[0];
        viewsDescs[1].texture = CTX.lobbyTextures[1];
    }
    EGLCHK("before CardboardDistortionRenderer_renderEyeToDisplay()");
    CardboardDistortionRenderer_renderEyeToDisplay(CTX.distortionRenderer,
                                                   0,
                                                   0,
                                                   0,
                                                   CTX.screenWidth,
                                                   CTX.screenHeight,
                                                   &viewsDescs[0],
                                                   &viewsDescs[1]);
    EGLCHK("after CardboardDistortionRenderer_renderEyeToDisplay()");
}

extern "C" JNIEXPORT void JNICALL
Java_viritualisres_phonevr_ALVRActivity_switchViewerNative(JNIEnv *, jobject) {
    CardboardQrCode_scanQrCodeAndSaveDeviceParams();
}
