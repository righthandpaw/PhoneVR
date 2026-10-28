#ifndef PHONEVR_MARKER_DETECTOR_H
#define PHONEVR_MARKER_DETECTOR_H

// Thin wrapper around the AprilTag detector (tag36h11 family). Platform independent: give it a
// grey (luma) image, get the tags back. Used by marker_spike.h on Android and by the desktop test.

#include <cmath>
#include <cstdint>
#include <vector>

#include "apriltag.h"
#include "tag36h11.h"

namespace phonevr {

struct MarkerDetection {
    int id = -1;
    float centre[2] = {0, 0};      // pixels
    float corners[4][2] = {};      // pixels, in AprilTag's winding order
    float sizePx = 0;              // mean edge length in pixels
    float margin = 0;              // AprilTag decision margin (bigger = more certain)
    int hamming = 0;               // corrected bit errors
};

class MarkerDetector {
public:
    struct Config {
        int threads = 2;
        // Detect on an image shrunk by this factor. 2 is about 3x faster than 1 and keeps tags
        // down to roughly 25-30 px wide (in the full-size image).
        float decimate = 2.0f;
        float blur = 0.0f;         // gaussian sigma; 0 = off
        bool refineEdges = true;
        int maxBitsCorrected = 1;  // tag36h11 can correct 1 bit error
    };

    MarkerDetector() : MarkerDetector(Config()) {}
    explicit MarkerDetector(const Config &cfg) {
        family_ = tag36h11_create();
        det_ = apriltag_detector_create();
        apriltag_detector_add_family_bits(det_, family_, cfg.maxBitsCorrected);
        det_->nthreads = cfg.threads;
        det_->quad_decimate = cfg.decimate;
        det_->quad_sigma = cfg.blur;
        det_->refine_edges = cfg.refineEdges ? 1 : 0;
    }
    ~MarkerDetector() {
        apriltag_detector_destroy(det_);
        tag36h11_destroy(family_);
    }
    MarkerDetector(const MarkerDetector &) = delete;
    MarkerDetector &operator=(const MarkerDetector &) = delete;

    // luma: 8-bit grey image, row stride in bytes. Fills `out` (cleared first).
    void detect(const uint8_t *luma, int width, int height, int stride,
                std::vector<MarkerDetection> &out) {
        out.clear();
        image_u8_t im = {width, height, stride, const_cast<uint8_t *>(luma)};
        zarray_t *dets = apriltag_detector_detect(det_, &im);
        for (int i = 0; i < zarray_size(dets); i++) {
            apriltag_detection_t *d;
            zarray_get(dets, i, &d);
            MarkerDetection m;
            m.id = d->id;
            m.centre[0] = (float) d->c[0];
            m.centre[1] = (float) d->c[1];
            float perimeter = 0;
            for (int k = 0; k < 4; k++) {
                m.corners[k][0] = (float) d->p[k][0];
                m.corners[k][1] = (float) d->p[k][1];
                int n = (k + 1) % 4;
                perimeter += std::hypot((float) (d->p[n][0] - d->p[k][0]),
                                        (float) (d->p[n][1] - d->p[k][1]));
            }
            m.sizePx = perimeter / 4.f;
            m.margin = d->decision_margin;
            m.hamming = d->hamming;
            out.push_back(m);
        }
        apriltag_detections_destroy(dets);
    }

private:
    apriltag_detector_t *det_ = nullptr;
    apriltag_family_t *family_ = nullptr;
};

}  // namespace phonevr

#endif  // PHONEVR_MARKER_DETECTOR_H
