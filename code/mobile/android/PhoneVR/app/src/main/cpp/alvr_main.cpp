#include "alvr_client_core.h"
#include "arcore_c_api.h"
#include "cardboard.h"
#include <GLES3/gl3.h>
#include <EGL/egl.h>
#include <algorithm>
#include <android/log.h>
#include <deque>
#include <jni.h>
#include <map>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

#include "nlohmann/json.hpp"
#include "utils.h"
#include <cmath>

using namespace nlohmann;

uint64_t HEAD_ID = alvr_path_string_to_id("/user/head");

// Keep this false: ARCore is used for position and slow orientation correction.
// The experimental fusion combines Cardboard's low-latency orientation
// with ARCore's absolute orientation without directly using ARCore's delayed pose.
bool useARCoreOrientation = false;

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

    bool arcoreEnabled = false;
    ArSession *arSession = nullptr;
    ArFrame *arFrame = nullptr;
    ArAnchor *arFloorAnchor = nullptr;
    GLuint arTexture = 0;

    AlvrQuat lastOrientation = {0.f, 0.f, 0.f, 0.f};
    AlvrQuat latestARCoreOrientation = {};
    float lastPosition[3] = {0.f, 0.f, 0.f};

    // Experimental orientation fusion state.
    // Cardboard remains the fast orientation source; ARCore slowly corrects
    // accumulated drift. This correction is currently diagnostic only.
    AlvrQuat orientationCorrection = {0.f, 0.f, 0.f, 1.f};
    uint64_t lastOrientationCorrectionNs = 0;

    AlvrQuat calibrationSum = {0.f, 0.f, 0.f, 0.f};
    int calibrationSamples = 0;

    // The ARCore pose must be acquired on the OpenGL/render thread.
    // inputThread has no EGL context, so it consumes the latest pose
    // produced by renderNative().
    AlvrPose latestPose = {};
    std::mutex poseMutex;
    bool latestPoseValid = false;

    int screenWidth = 0;
    int screenHeight = 0;

    bool renderingParamsChanged = true;
    bool glContextRecreated = false;

    bool running = false;
    bool streaming = false;
    float displayRefreshRate = 60.0f;
    std::thread inputThread;

    // Une one texture per eye, no need for swapchains.
    GLuint lobbyTextures[2] = {0, 0};
    GLuint streamTextures[2] = {0, 0};

    float eyeOffsets[2] = {0.0, 0.0};
    AlvrFov fovArr[2] = {};
    AlvrViewParams viewParams[2] = {};
    AlvrDeviceMotion deviceMotion = {};
    
    // Store last decoded frame for reprojection when no new frame arrives
    void *lastStreamBuffer = nullptr;
    int64_t lastStreamTimestampNs = -1;

    NativeContext() {
        // memset(&fovArr, 0, (sizeof(fovArr)) / sizeof(int));
       // memset(&viewParams, 0, (sizeof(viewParams)) / sizeof(int));
       // memset(&deviceMotion, 0, (sizeof(deviceMotion)) / sizeof(int));
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

//AI Maths

static AlvrQuat quatMultiply(const AlvrQuat& a, const AlvrQuat& b) {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    };
}

static AlvrQuat quatInverse(const AlvrQuat& q) {
    // Assumes q is normalized.
    return {-q.x, -q.y, -q.z, q.w};
}

static float quatAngleDegrees(const AlvrQuat& q) {
    // q and -q represent the same rotation, so use abs(w).
    float w = std::clamp(std::abs(q.w), 0.0f, 1.0f);
    return 2.0f * std::acos(w) * 180.0f / M_PI;
}

AlvrQuat normalizeQuat(AlvrQuat q) {
    float len = std::sqrt(
        q.x * q.x +
        q.y * q.y +
        q.z * q.z +
        q.w * q.w);

    if (len < 1e-6f)
        return {0.f, 0.f, 0.f, 1.f};

    return {
        q.x / len,
        q.y / len,
        q.z / len,
        q.w / len
    };
}

AlvrQuat quatScale(AlvrQuat q, float s) {
    return {
        q.x * s,
        q.y * s,
        q.z * s,
        q.w * s
    };
}

AlvrQuat quatAdd(AlvrQuat a, AlvrQuat b) {
    return {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
        a.w + b.w
    };
}

AlvrQuat slerpQuat(AlvrQuat a, AlvrQuat b, float t) {
    a = normalizeQuat(a);
    b = normalizeQuat(b);

    float dot =
        a.x * b.x +
        a.y * b.y +
        a.z * b.z +
        a.w * b.w;

    // Take the shortest path.
    if (dot < 0.0f) {
        b.x = -b.x;
        b.y = -b.y;
        b.z = -b.z;
        b.w = -b.w;
        dot = -dot;
    }

    // Very close: normalized linear interpolation is sufficient.
    if (dot > 0.9995f) {
        AlvrQuat result = {
            a.x + t * (b.x - a.x),
            a.y + t * (b.y - a.y),
            a.z + t * (b.z - a.z),
            a.w + t * (b.w - a.w)
        };

        return normalizeQuat(result);
    }

    float theta = std::acos(dot);
    float sinTheta = std::sin(theta);

    float wa = std::sin((1.0f - t) * theta) / sinTheta;
    float wb = std::sin(t * theta) / sinTheta;

    return normalizeQuat({
        wa * a.x + wb * b.x,
        wa * a.y + wb * b.y,
        wa * a.z + wb * b.z,
        wa * a.w + wb * b.w
    });
}

//


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

AlvrPose getPose(uint64_t timestampNs) {
    AlvrPose pose = {};
    bool returnLastPosition = false;

    if (!CTX.arcoreEnabled || (CTX.arcoreEnabled && !useARCoreOrientation)) {
        float pos[3];
        float q[4];
        CardboardHeadTracker_getPose(CTX.headTracker, (int64_t) timestampNs, kLandscapeLeft, pos, q);

        auto inverseOrientation = AlvrQuat{q[0], q[1], q[2], q[3]};
        pose.orientation = inverseQuat(inverseOrientation);
        CTX.lastOrientation = pose.orientation;
    }

    if (CTX.arcoreEnabled && CTX.arSession != nullptr) {

        if (CTX.arTexture == 0) {
            returnLastPosition = true;
            goto out;
        }

        // getPose() must only be called from the OpenGL/render thread.
        // Do not attempt to recover from a missing EGL context here.
        // inputThread never calls getPose().
        if (eglGetCurrentContext() == EGL_NO_CONTEXT) {
            error("getPose: BUG - called without an EGL context");
            returnLastPosition = true;
            goto out;
        }

        int ret = ArSession_update(CTX.arSession, CTX.arFrame);
        if (ret != AR_SUCCESS) {
            error("getPose: ArSession_update failed (%d), using last position", ret);
            returnLastPosition = true;
            goto out;
        }

        ArCamera *arCamera = nullptr;
        ArFrame_acquireCamera(CTX.arSession, CTX.arFrame, &arCamera);

        ArTrackingState arTrackingState;
        ArCamera_getTrackingState(CTX.arSession, arCamera, &arTrackingState);
        if (arTrackingState != AR_TRACKING_STATE_TRACKING) {
            error("getPose: Camera is not tracking, using last position");
            if (arTrackingState == AR_TRACKING_STATE_PAUSED) {
                error("- AR tracking state is PAUSED");
                ArTrackingFailureReason failureReason;
                ArCamera_getTrackingFailureReason(CTX.arSession, arCamera, &failureReason);
                error("- Failure reason: %d", failureReason);
            } else if (arTrackingState == AR_TRACKING_STATE_STOPPED) {
                error("- AR tracking state is STOPPED");
            }
            returnLastPosition = true;
            ArCamera_release(arCamera);
            goto out;
        }

        ArPose *arPose = nullptr;
        ArPose_create(CTX.arSession, nullptr, &arPose);
        ArCamera_getDisplayOrientedPose(CTX.arSession, arCamera, arPose);
        // ArPose_getPoseRaw() returns a pose in {qx, qy, qz, qw, tx, ty, tz} format.
        float arRawPose[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        ArPose_getPoseRaw(CTX.arSession, arPose, arRawPose);

        /* We determine floor position by finding the lowest detected plane. We do this by
         * placing an anchor in the center position of the plane, and if it's lower than the
         * currently placed anchor (CTX.floorAnchor), we replace it.
         *
         * (By default, ARCore's "world coordinates" space begins wherever the device is, but this
         * can desync over time. Anchor position adapts to world space movement. */
        ArTrackableList *trackables = nullptr;
        ArTrackableList_create(CTX.arSession, &trackables);
        ArFrame_getUpdatedTrackables(CTX.arSession, CTX.arFrame, AR_TRACKABLE_PLANE, trackables);
        int32_t detectedPlaneCount;
        ArTrackableList_getSize(CTX.arSession, trackables, &detectedPlaneCount);

        for (int i = 0; i < detectedPlaneCount; i++) {
            ArTrackable* arTrackable = nullptr;
            ArTrackableList_acquireItem(CTX.arSession, trackables, i,
                                        &arTrackable);
            const ArPlane *plane = ArAsPlane(arTrackable);
            ArTrackingState planeTrackingState;
            ArTrackable_getTrackingState(CTX.arSession, arTrackable, &planeTrackingState);
            ArPlaneType planeType;
            ArPlane_getType(CTX.arSession, plane, &planeType);
            if (planeTrackingState == AR_TRACKING_STATE_TRACKING &&
                planeType == AR_PLANE_HORIZONTAL_UPWARD_FACING) {
                ArPose *planePose = nullptr;
                ArPose_create(CTX.arSession, nullptr, &planePose);
                ArPlane_getCenterPose(CTX.arSession, plane, planePose);

                bool reanchor = false;
                if (CTX.arFloorAnchor == nullptr) {
                    reanchor = true;
                } else {
                    ArPose *currentFloorPose = nullptr;
                    ArPose_create(CTX.arSession, nullptr, &currentFloorPose);

                    ArAnchor_getPose(CTX.arSession, CTX.arFloorAnchor, currentFloorPose);
                    float currentFloorPoseRaw[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    ArPose_getPoseRaw(CTX.arSession, currentFloorPose, currentFloorPoseRaw);

                    float planePoseRaw[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    ArPose_getPoseRaw(CTX.arSession, planePose, planePoseRaw);

                    if (planePoseRaw[5] < currentFloorPoseRaw[5]) {
                        info("Found new plane lower than pose (current %f vs new %f), reanchoring",
                             currentFloorPoseRaw[5], planePoseRaw[5]);
                        reanchor = true;
                    }
                }

                if (reanchor) {
                    if (CTX.arFloorAnchor != nullptr) {
                        ArAnchor_detach(CTX.arSession, CTX.arFloorAnchor);
                        ArAnchor_release(CTX.arFloorAnchor);
                    }

                    ArPose *planePoseNoRotation = ArPose_extractTranslation(CTX.arSession,
                                                                            planePose);
                    ArTrackable_acquireNewAnchor(CTX.arSession, arTrackable, planePoseNoRotation,
                                                 &CTX.arFloorAnchor);
                    ArPose_destroy(planePoseNoRotation);
                }

                ArPose_destroy(planePose);
            }
        }

        ArTrackableList_destroy(trackables);

        float anchorRawPose[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (CTX.arFloorAnchor != nullptr) {
            ArPose *anchorPose = nullptr;
            ArPose_create(CTX.arSession, nullptr, &anchorPose);
            ArAnchor_getPose(CTX.arSession, CTX.arFloorAnchor, anchorPose);
            ArPose_getPoseRaw(CTX.arSession, anchorPose, anchorRawPose);
            info("anchor pose %f %f %f %f %f %f %f", anchorRawPose[0], anchorRawPose[1], anchorRawPose[2], anchorRawPose[3], anchorRawPose[4], anchorRawPose[5], anchorRawPose[6]);
        }

        pose.position[0] = arRawPose[4];
        pose.position[1] = arRawPose[5] - anchorRawPose[5];
        pose.position[2] = arRawPose[6];

        for (int i = 0; i < 3; i++) {
            CTX.lastPosition[i] = arRawPose[i + 4];
        }

        /*
        if (useARCoreOrientation) {
            
            auto orientation = AlvrQuat{arRawPose[0], arRawPose[1], arRawPose[2],
                                                arRawPose[3]};
            pose.orientation = orientation;
            CTX.lastOrientation = pose.orientation;
            
            
        }
        */

        auto arcoreOrientation = AlvrQuat{
                arRawPose[0],
                arRawPose[1],
                arRawPose[2],
                arRawPose[3]
            };

        // DEBUG ONLY.
        // Store the raw ARCore orientation for comparison.
        // This does NOT affect the pose returned by getPose().
        CTX.latestARCoreOrientation = arcoreOrientation;

        if (useARCoreOrientation) {
            pose.orientation = arcoreOrientation;
            CTX.lastOrientation = pose.orientation;
        }


        ArPose_destroy(arPose);
        ArCamera_release(arCamera);
    }

out:
    if (returnLastPosition) {
        pose.orientation = CTX.lastOrientation;
        for (int i = 0; i < 3; i++) {
            pose.position[i] = CTX.lastPosition[i];
        }
    }

    return pose;
}

void updateViewConfigs(uint64_t targetTimestampNs = 0) {
    if (!targetTimestampNs)
        targetTimestampNs = GetBootTimeNano() + alvr_get_head_prediction_offset_ns();

    AlvrPose headPose = {};

    {
        std::lock_guard<std::mutex> lock(CTX.poseMutex);

        if (!CTX.latestPoseValid) {
            error("updateViewConfigs: No ARCore pose available yet");
            return;
        }

        headPose = CTX.latestPose;
    }

    CTX.deviceMotion.device_id = HEAD_ID;
    CTX.deviceMotion.pose = headPose;

    float headToEye[3] = {CTX.eyeOffsets[kLeft], 0.0, 0.0};

    CTX.viewParams[kLeft].pose = headPose;
    offsetPosWithQuat(headPose.orientation,
                      headToEye,
                      CTX.viewParams[kLeft].pose.position);
    CTX.viewParams[kLeft].fov = CTX.fovArr[kLeft];

    headToEye[0] = CTX.eyeOffsets[kRight];
    CTX.viewParams[kRight].pose = headPose;
    offsetPosWithQuat(headPose.orientation,
                      headToEye,
                      CTX.viewParams[kRight].pose.position);
    CTX.viewParams[kRight].fov = CTX.fovArr[kRight];
}

void inputThread() {
    auto deadline = std::chrono::steady_clock::now();

    info("inputThread: thread staring...");
    while (CTX.streaming) {

        auto targetTimestampNs = GetBootTimeNano() + alvr_get_head_prediction_offset_ns();
        updateViewConfigs(targetTimestampNs);

        alvr_send_tracking(
            targetTimestampNs, CTX.viewParams, &CTX.deviceMotion, 1, nullptr, nullptr);

        deadline += std::chrono::nanoseconds((uint64_t) (1e9 / CTX.displayRefreshRate / 3));
        std::this_thread::sleep_until(deadline);
    }
}

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
    CTX.javaVm = vm;
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL
Java_viritualisres_phonevr_ALVRActivity_initializeNative(
        JNIEnv *env,
        jobject obj,
        jint screenWidth,
        jint screenHeight,
        jfloat refreshRate,
        jboolean enableARCore) {

    CTX.javaContext = env->NewGlobalRef(obj);

    uint32_t viewWidth = std::max(screenWidth, screenHeight) / 2;
    uint32_t viewHeight = std::min(screenWidth, screenHeight);

    // Save display information for ARCore / rendering.
    CTX.screenWidth = screenWidth;
    CTX.screenHeight = screenHeight;
    CTX.displayRefreshRate = refreshRate;

    alvr_initialize_android_context(
            (void *) CTX.javaVm,
            (void *) CTX.javaContext);

    float refreshRatesBuffer[1] = {refreshRate};

    AlvrClientCapabilities caps = {};
    caps.default_view_height = viewHeight;
    caps.default_view_width = viewWidth;
    caps.external_decoder = false;
    caps.refresh_rates = refreshRatesBuffer;
    caps.refresh_rates_count = 1;
    caps.foveated_encoding = true;
    caps.encoder_high_profile = true;
    caps.encoder_10_bits = true;
    caps.encoder_av1 = true;

    alvr_initialize(caps);

    Cardboard_initializeAndroid(CTX.javaVm, CTX.javaContext);
    CTX.headTracker = CardboardHeadTracker_create();

    CTX.arcoreEnabled = (bool) enableARCore;

    if (CTX.arcoreEnabled) {
        if (ArSession_create(env, CTX.javaContext, &CTX.arSession) != AR_SUCCESS) {
            error("initializeNative: Could not create ARCore session");
            CTX.arcoreEnabled = false;
            return;
        }

        ArConfig *arConfig = nullptr;
        ArConfig_create(CTX.arSession, &arConfig);

        // Explicitly disable all unnecessary features to preserve CPU power.
        ArConfig_setDepthMode(
                CTX.arSession,
                arConfig,
                AR_DEPTH_MODE_DISABLED);

        ArConfig_setLightEstimationMode(
                CTX.arSession,
                arConfig,
                AR_LIGHT_ESTIMATION_MODE_DISABLED);

        ArConfig_setPlaneFindingMode(
                CTX.arSession,
                arConfig,
                AR_PLANE_FINDING_MODE_HORIZONTAL_AND_VERTICAL);

        ArConfig_setCloudAnchorMode(
                CTX.arSession,
                arConfig,
                AR_CLOUD_ANCHOR_MODE_DISABLED);

        // Don't block waiting for a new camera image.
        ArConfig_setUpdateMode(
                CTX.arSession,
                arConfig,
                AR_UPDATE_MODE_LATEST_CAMERA_IMAGE);

        if (ArSession_configure(CTX.arSession, arConfig) != AR_SUCCESS) {
            error("initializeNative: Could not configure ARCore session");
            return;
        }

        ArFrame_create(CTX.arSession, &CTX.arFrame);
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
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_resumeNative(JNIEnv *,
                                                                                       jobject) {
    CardboardHeadTracker_resume(CTX.headTracker);

    if (CTX.arcoreEnabled && CTX.arSession != nullptr) {
        ArStatus arSessionStatus = ArSession_resume(CTX.arSession);
        if (arSessionStatus != AR_SUCCESS) {
            error("Failed to resume tracking: %d", arSessionStatus);
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
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_pauseNative(JNIEnv *,
                                                                                      jobject) {
    alvr_pause();

    if (CTX.running) {
        CTX.running = false;
    }

    CardboardHeadTracker_pause(CTX.headTracker);
}

extern "C" JNIEXPORT void JNICALL
Java_viritualisres_phonevr_ALVRActivity_surfaceCreatedNative(JNIEnv *, jobject) {
    alvr_initialize_opengl();

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
    try {
        if (CTX.renderingParamsChanged) {
            info("renderingParamsChanged, processing new params");
            uint8_t *buffer;
            int size;
            CardboardQrCode_getSavedDeviceParams(&buffer, &size);

            if (size == 0) {
				CardboardQrCode_destroy(buffer);  // must free even when empty
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

            info("renderingParamsChanged, updating new view configs (FOV) to alvr");
            // alvr_send_views_config(fovArr, CTX.eyeOffsets[0] - CTX.eyeOffsets[1]);
        }

        // Note: if GL context is recreated, old resources are already freed.
        if (CTX.renderingParamsChanged && !CTX.glContextRecreated) {
            info("Pausing ALVR since glContext is not recreated, deleting textures");
            alvr_pause_opengl();

            GL(glDeleteTextures(2, CTX.lobbyTextures));
        }

        if (CTX.renderingParamsChanged || CTX.glContextRecreated) {
            info("Rebuilding, binding textures, Resuming ALVR since glContextRecreated %b, "
                 "renderingParamsChanged %b",
                 CTX.renderingParamsChanged,
                 CTX.glContextRecreated);
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

            const uint32_t *targetViews[2] = {(uint32_t *) &CTX.lobbyTextures[0],
                                              (uint32_t *) &CTX.lobbyTextures[1]};
            alvr_resume_opengl(CTX.screenWidth / 2, CTX.screenHeight, targetViews, 1, true);

            // Initialize ARCore camera texture after the OpenGL context is ready.
            if (CTX.arcoreEnabled && CTX.arSession != nullptr) {
                GLuint arTextureIdArray[1];
                glGenTextures(1, arTextureIdArray);
                CTX.arTexture = arTextureIdArray[0];

                GL(glBindTexture(GL_TEXTURE_2D, CTX.arTexture));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

                ArSession_setCameraTextureName(CTX.arSession, CTX.arTexture);

                info("ARCore camera texture initialized: %u", CTX.arTexture);
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

                if (message_length > 0)
                    alvr_update_hud_message_opengl(&message_buffer[0]);
            }
            if (event.tag == ALVR_EVENT_STREAMING_STARTED) {
                info("ALVR Poll Event: ALVR_EVENT_STREAMING_STARTED, generating and binding "
                     "textures...");
                auto config = event.STREAMING_STARTED;

                auto settings_len = alvr_get_settings_json(nullptr);
                auto settings_buffer = std::vector<char>(settings_len);
                alvr_get_settings_json(&settings_buffer[0]);

                info("Got settings from ALVR Server - %s", &settings_buffer[0]);
                if (settings_len > 900)   // to workthough logcat buffer limit
                    info("Got settings from ALVR Server - %s", &settings_buffer[900]);
                json settings_json = json::parse(&settings_buffer[0]);

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
                                    config.view_width,
                                    config.view_height,
                                    0,
                                    GL_RGB,
                                    GL_UNSIGNED_BYTE,
                                    nullptr));
                }

                CTX.fovArr[0] = getFov((CardboardEye) 0);
                CTX.fovArr[1] = getFov((CardboardEye) 1);

                info("ALVR Poll Event: ALVR_EVENT_STREAMING_STARTED, View configs updated...");

                auto leftIntHandle = (uint32_t) CTX.streamTextures[0];
                auto rightIntHandle = (uint32_t) CTX.streamTextures[1];
                const uint32_t *textureHandles[2] = {&leftIntHandle, &rightIntHandle};

                auto render_config = AlvrStreamConfig{};
                render_config.view_resolution_width = config.view_width;
                render_config.view_resolution_height = config.view_height;
                render_config.swapchain_textures = textureHandles;
                render_config.swapchain_length = 1;

                render_config.enable_foveation = false;
                if (!settings_json["video"].is_null()) {
                    if (!settings_json["video"]["foveated_encoding"].is_null()) {
                        info("settings_json.video.foveated_encoding is %s",
                             settings_json["video"]["foveated_encoding"].dump().c_str());

                        // Foveated encoding would be a "Enabled": {Array} or "Disabled" String
                        if (!settings_json["video"]["foveated_encoding"].is_string()) {
                            render_config.enable_foveation = true;
                            render_config.foveation_center_size_x =
                                settings_json["video"]["foveated_encoding"]["Enabled"]
                                             ["center_size_x"];
                            render_config.foveation_center_size_y =
                                settings_json["video"]["foveated_encoding"]["Enabled"]
                                             ["center_size_y"];
                            render_config.foveation_center_shift_x =
                                settings_json["video"]["foveated_encoding"]["Enabled"]
                                             ["center_shift_x"];
                            render_config.foveation_center_shift_y =
                                settings_json["video"]["foveated_encoding"]["Enabled"]
                                             ["center_shift_y"];
                            render_config.foveation_edge_ratio_x =
                                settings_json["video"]["foveated_encoding"]["Enabled"]
                                             ["edge_ratio_x"];
                            render_config.foveation_edge_ratio_y =
                                settings_json["video"]["foveated_encoding"]["Enabled"]
                                             ["edge_ratio_y"];
                        } else
                            info("foveated_encoding is Disabled");
                    } else
                        error("settings_json doesn't have a video.foveated_encoding key");
                } else
                    error("settings_json doesn't have a video key");

                info("Settings for foveation:");
                info("render_config.enable_foveation: %b", render_config.enable_foveation);
                info("render_config.foveation_center_size_x: %f",
                     render_config.foveation_center_size_x);
                info("render_config.foveation_center_size_y: %f",
                     render_config.foveation_center_size_y);
                info("render_config.foveation_center_shift_x: %f",
                     render_config.foveation_center_shift_x);
                info("render_config.foveation_center_shift_y: %f",
                     render_config.foveation_center_shift_y);
                info("render_config.foveation_edge_ratio_x: %f",
                     render_config.foveation_edge_ratio_x);
                info("render_config.foveation_edge_ratio_y: %f",
                     render_config.foveation_edge_ratio_y);

                alvr_start_stream_opengl(render_config);

                // Start each streaming session with no accumulated orientation
                // correction. Cardboard remains the immediate orientation source.
                CTX.orientationCorrection = {0.f, 0.f, 0.f, 1.f};
                CTX.lastOrientationCorrectionNs = 0;

                info("ALVR Poll Event: ALVR_EVENT_STREAMING_STARTED, opengl stream started and "
                     "input "
                     "Thread started...");
                CTX.streaming = true;
                CTX.inputThread = std::thread(inputThread);

            } else if (event.tag == ALVR_EVENT_STREAMING_STOPPED) {
                info("ALVR Poll Event: ALVR_EVENT_STREAMING_STOPPED, Waiting for inputThread to "
                     "join...");
                CTX.lastStreamBuffer = nullptr;   // clear reprojection buffer on disconnect
                CTX.lastStreamTimestampNs = -1;
                CTX.streaming = false;
                CTX.inputThread.join();

                GL(glDeleteTextures(2, CTX.streamTextures));
                CTX.streamTextures[0] = 0;  // zero out so stale handles can't be used
                CTX.streamTextures[1] = 0;
                info("ALVR Poll Event: ALVR_EVENT_STREAMING_STOPPED, Stream stopped deleted "
                     "textures.");
            }
        }

        CardboardEyeTextureDescription viewsDescs[2] = {};
        for (auto &viewsDesc : viewsDescs) {
            viewsDesc.left_u = 0.0;
            viewsDesc.right_u = 1.0;
            viewsDesc.top_v = 1.0;
            viewsDesc.bottom_v = 0.0;
        }

        if (CTX.streaming) {

            // ARCore requires an EGL context. renderNative() runs on the
            // OpenGL/render thread, so this is the correct place to update it.
            if (CTX.arcoreEnabled && CTX.arSession != nullptr) {
                uint64_t poseTimestampNs =
                        GetBootTimeNano() + alvr_get_head_prediction_offset_ns();

                AlvrPose pose = getPose(poseTimestampNs);

                {
                    std::lock_guard<std::mutex> lock(CTX.poseMutex);
                    CTX.latestPose = pose;
                    CTX.latestPoseValid = true;
                }
            }

            // EXPERIMENTAL ORIENTATION FUSION.
            //
            // Cardboard remains the fast, low-latency orientation source.
            // ARCore is used only as a slow absolute reference.
            //
            // The fused orientation is used for actual tracking below.
            // ARCore position remains unchanged; only orientation is replaced.
            if (CTX.arcoreEnabled && CTX.arSession != nullptr) {
                uint64_t nowNs = GetBootTimeNano();

                // Get Cardboard's current orientation using the same predicted
                // timestamp style as the actual tracking path.
                float cardboardPos[3] = {};
                float cardboardQ[4] = {};

                uint64_t cardboardTimestampNs =
                    nowNs + alvr_get_head_prediction_offset_ns();

                CardboardHeadTracker_getPose(
                    CTX.headTracker,
                    (int64_t) cardboardTimestampNs,
                    kLandscapeLeft,
                    cardboardPos,
                    cardboardQ);

                AlvrQuat cardboardRawOrientation = {
                    cardboardQ[0],
                    cardboardQ[1],
                    cardboardQ[2],
                    cardboardQ[3]
                };

                // This is the orientation actually used by PhoneVR when
                // useARCoreOrientation == false.
                AlvrQuat cardboardOrientation =
                    normalizeQuat(inverseQuat(cardboardRawOrientation));

                // getPose() has already updated this on the render thread.
                AlvrQuat arcoreOrientation =
                    normalizeQuat(CTX.latestARCoreOrientation);

                // Fixed coordinate-frame conversion established during
                // calibration. This converts ARCore into Cardboard's frame.
                const AlvrQuat arcoreToCardboard = {
                    -0.0057f,
                    -0.0031f,
                     0.7076f,
                     0.7066f
                };

                AlvrQuat arcoreInCardboard =
                    normalizeQuat(
                        quatMultiply(
                            arcoreOrientation,
                            quatInverse(arcoreToCardboard)));

                // Rotation that takes the current Cardboard orientation
                // toward the ARCore orientation.
                AlvrQuat relativeOrientation =
                    normalizeQuat(
                        quatMultiply(
                            quatInverse(cardboardOrientation),
                            arcoreInCardboard));

                float errorAngle = quatAngleDegrees(relativeOrientation);

                float correctionDt = 0.0f;
                float correctionGainUsed = 0.0f;
                bool correctionAccepted = false;

                // Update the correction every render frame, but make the
                // amount of correction depend on elapsed time rather than
                // frame rate.
                //
                // A 3-second time constant means:
                //   gain = 1 - exp(-dt / 3)
                //
                // This deliberately keeps Cardboard dominant during rapid
                // head movement while slowly removing long-term drift.
                if (errorAngle < 15.0f) {
                    correctionAccepted = true;
                    if (CTX.lastOrientationCorrectionNs == 0) {
                        CTX.lastOrientationCorrectionNs = nowNs;
                    }

                    uint64_t deltaNs =
                        nowNs - CTX.lastOrientationCorrectionNs;

                    // Protect against an unexpectedly large gap (pause,
                    // debugger, lifecycle transition, etc.).
                    if (deltaNs > 0 && deltaNs < 500000000ULL) {
                        float dt = static_cast<float>(deltaNs) * 1e-9f;
                        correctionDt = dt;

                        constexpr float correctionTimeConstant = 3.0f;
                        float correctionGain =
                            1.0f -
                            std::exp(-dt / correctionTimeConstant);
                        correctionGainUsed = correctionGain;

                        CTX.orientationCorrection =
                            slerpQuat(
                                CTX.orientationCorrection,
                                relativeOrientation,
                                correctionGain);
                    }

                    CTX.lastOrientationCorrectionNs = nowNs;
                } else {
                    // Do not allow a transient ARCore/Cardboard mismatch to
                    // pull the correction toward a bad measurement.
                    CTX.lastOrientationCorrectionNs = nowNs;
                }

                // This is what we would eventually feed into latestPose.
                // For now it is deliberately NOT used by PhoneVR.
                AlvrQuat fusedOrientation =
                    normalizeQuat(
                        quatMultiply(
                            cardboardOrientation,
                            CTX.orientationCorrection));

                AlvrQuat fusedError =
                    normalizeQuat(
                        quatMultiply(
                            quatInverse(fusedOrientation),
                            arcoreInCardboard));

                float fusedErrorAngle =
                    quatAngleDegrees(fusedError);

                // Enable the fused orientation for actual headset tracking.
                // latestPose already contains the ARCore position from getPose(),
                // so this changes rotation only.
                {
                    std::lock_guard<std::mutex> lock(CTX.poseMutex);
                    if (CTX.latestPoseValid) {
                        CTX.latestPose.orientation = fusedOrientation;
                    }
                }

                // Log at approximately 1 Hz so logcat remains manageable.
                static uint64_t lastOrientationLogNs = 0;
                if (nowNs - lastOrientationLogNs >= 1000000000ULL) {
                    lastOrientationLogNs = nowNs;

                    info(
                        "FUSION TEST: "
                        "ARCore error=%.2f deg "
                        "correction=%.2f deg "
                        "fused error=%.2f deg",
                        errorAngle,
                        quatAngleDegrees(CTX.orientationCorrection),
                        fusedErrorAngle
                    );

                    info(
                        "FUSION PARAMS: "
                        "dt=%.3f ms "
                        "gain=%.5f "
                        "accepted=%s",
                        correctionDt * 1000.0f,
                        correctionGainUsed,
                        correctionAccepted ? "yes" : "no"
                    );

                    info(
                        "ORIENTATION ERROR: "
                        "angle=%.3f deg "
                        "error=[%.5f, %.5f, %.5f, %.5f]",
                        errorAngle,
                        relativeOrientation.x,
                        relativeOrientation.y,
                        relativeOrientation.z,
                        relativeOrientation.w
                    );

                    info(
                        "ORIENTATION DEBUG:\n"
                        "  Cardboard raw: [%.5f, %.5f, %.5f, %.5f]\n"
                        "  Cardboard used: [%.5f, %.5f, %.5f, %.5f]\n"
                        "  ARCore:         [%.5f, %.5f, %.5f, %.5f]\n"
                        "  Relative:       [%.5f, %.5f, %.5f, %.5f]\n"
                        "  Difference angle: %.2f deg",
                        cardboardRawOrientation.x,
                        cardboardRawOrientation.y,
                        cardboardRawOrientation.z,
                        cardboardRawOrientation.w,

                        cardboardOrientation.x,
                        cardboardOrientation.y,
                        cardboardOrientation.z,
                        cardboardOrientation.w,

                        arcoreOrientation.x,
                        arcoreOrientation.y,
                        arcoreOrientation.z,
                        arcoreOrientation.w,

                        relativeOrientation.x,
                        relativeOrientation.y,
                        relativeOrientation.z,
                        relativeOrientation.w,

                        errorAngle
                    );
                }
            }

            void *streamHardwareBuffer = nullptr;

            AlvrViewParams dummyViewParams;
            auto timestampNs = alvr_get_frame(&dummyViewParams, &streamHardwareBuffer);

            if (timestampNs == -1) {
                // No new decoded frame this cycle.
                // Instead of returning (which causes a blank/ghost frame), reuse the
                // last valid buffer — this is basic async reprojection: same image,
                // but composited with the current head orientation by Cardboard.
                if (CTX.lastStreamBuffer == nullptr) {
                    // We truly have no frame yet at all, nothing to show.
                    return;
                }
                streamHardwareBuffer = CTX.lastStreamBuffer;
                timestampNs = CTX.lastStreamTimestampNs;
            } else {
                // New frame arrived — save it for future reprojection cycles.
                CTX.lastStreamBuffer = streamHardwareBuffer;
                CTX.lastStreamTimestampNs = timestampNs;
            }

            uint32_t swapchainIndices[2] = {0, 0};
            alvr_render_stream_opengl(streamHardwareBuffer, swapchainIndices);

            alvr_report_submit(timestampNs, 0);

            viewsDescs[0].texture = CTX.streamTextures[0];
            viewsDescs[1].texture = CTX.streamTextures[1];
        } else {
            AlvrPose pose = getPose(GetBootTimeNano() + VSYNC_QUEUE_INTERVAL_NS);

            {
                std::lock_guard<std::mutex> lock(CTX.poseMutex);
                CTX.latestPose = pose;
                CTX.latestPoseValid = true;
            }

            AlvrViewInput viewInputs[2] = {};
            for (int eye = 0; eye < 2; eye++) {
                float headToEye[3] = {CTX.eyeOffsets[eye], 0.0, 0.0};
                // offset head pos to Eye Position
                offsetPosWithQuat(pose.orientation, headToEye, viewInputs[eye].pose.position);

                viewInputs[eye].pose.orientation = pose.orientation;
                viewInputs[eye].fov = getFov((CardboardEye) eye);
                viewInputs[eye].swapchain_index = 0;
            }
            alvr_render_lobby_opengl(viewInputs);

            viewsDescs[0].texture = CTX.lobbyTextures[0];
            viewsDescs[1].texture = CTX.lobbyTextures[1];
        }

        // Note: the Cardboard SDK does not support reprojection!
        // todo: manually implement it?

        // info("nativeRendered: Rendering to Display...");
        CardboardDistortionRenderer_renderEyeToDisplay(CTX.distortionRenderer,
                                                       0,
                                                       0,
                                                       0,
                                                       CTX.screenWidth,
                                                       CTX.screenHeight,
                                                       &viewsDescs[0],
                                                       &viewsDescs[1]);
    } catch (const json::exception &e) {
        error(std::string(std::string(__FUNCTION__) + std::string(__FILE_NAME__) +
                          std::string(e.what()))
                  .c_str());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_viritualisres_phonevr_ALVRActivity_switchViewerNative(JNIEnv *, jobject) {
    CardboardQrCode_scanQrCodeAndSaveDeviceParams();
}
