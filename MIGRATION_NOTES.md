# PhoneVR: ALVR v20 -> v21 migration notes

Generated while porting the Android client's `main` flavor (the one built on `alvr_client_core`,
not the legacy `gvr` custom-protocol flavor) from ALVR v20.8.0 to the v21 nightly
(commit `dc84d6835d4125226f9698f1f052629852d86e31`).

## 1. Pruned (removed)

- `code/mobile/android/PhoneVR/app/src/gvr/` and `app/src/androidTestGvr/` - the legacy,
  fully custom UDP-socket protocol client (`PVRSockets`/`PVRRenderer`, Google VR SDK). Superseded
  entirely by the `main` flavor, which already links `alvr_client_core` directly.
- `code/mobile/mobile-common/` - only consumed by the removed `gvr` flavor.
- `gvrImplementation 'com.google.vr:sdk-base:...'` and the GVR SDK unzip step in
  `extractNdk` (`app/build.gradle`).
- The `flavorDimensions "gvr"` / `productFlavors { gvr / noGvr }` split - now a single
  build variant.
- `IF(${HAVEGVR}) add_subdirectory(src/gvr) ENDIF()` in `CMakeLists.txt`, plus the now-unused
  `asio`/`eigen`/`ifaddrs` include dirs (only needed by the removed custom protocol).
- The GVR SDK download block in `prepare-alvr-deps.sh`, and the `nogvr` CLI flag it existed for.
- Manual settings-JSON parsing for foveated encoding in `alvr_main.cpp` - v21 delivers
  `foveated_encoding` directly on the `StreamingStarted` event, so the `nlohmann/json` dependency
  is gone from this file entirely.

**Not yet touched, but likely also obsolete now:** `driver/PVRServer` (a SteamVR driver for the
removed custom protocol's PC side) and `code/windows`. Left alone since they're outside the phone
client and weren't explicitly in scope. `.github/workflows/build.yml` and `release.yml` still
reference the deleted `gvr`/`noGvr` Gradle tasks (`assembleGvrDebug`, etc.) and will need
simplifying to a single variant before CI will pass again.

## 2. `alvr_client_core.h` (provisional)

This environment has no Rust toolchain or network access, so the real
`cargo xtask build-client-lib` -> `cbindgen` pipeline couldn't be run. The header at
`ALVR/build/alvr_client_core/alvr_client_core.h` was hand-derived by reading
`alvr/client_core/src/c_api.rs` and `alvr/common/src/c_api.rs` directly, following the project's
own `client_core/cbindgen.toml` (`QualifiedScreamingSnakeCase` enum tags, C-compatible layout).

**Before shipping a real build:** run `prepare-alvr-deps.sh` (which invokes
`cargo xtask build-client-lib`) against your ALVR nightly checkout and diff the generated header
against this one. The Gradle `buildClientLib` task overwrites this same path on every build, so
this hand-written version is only there to make the port compile/review before that's possible.

## 3. `alvr_main.cpp` API changes (v20 -> v21)

| Area | v20 | v21 |
|---|---|---|
| View params | Baked into `alvr_send_tracking`'s 2nd arg, recomputed every tick | Sent once via `alvr_send_view_params()` (head-local pose only), whenever eye offsets/FOV change |
| Head prediction offset | `alvr_get_head_prediction_offset_ns()` | **Removed from the C API.** Now a fixed heuristic (`ESTIMATED_PREDICTION_OFFSET_NS`, 30ms) - see the `TODO(upstream)` comment. Consider upstreaming a C export for `ClientCoreContext::get_total_prediction_offset()`. |
| `alvr_resume_opengl` | 5 args (trailing `always_direct` bool) | 4 args, bool dropped |
| `alvr_render_lobby_opengl` | `AlvrViewInput[2]` | `AlvrLobbyViewParams[2]` (adds `swapchain_index` wrapper) + `render_background: bool` |
| `alvr_render_stream_opengl` | `(buffer, swapchain_indices[2])` | `(buffer, AlvrStreamViewParams[2], foveation_center_shifts)` - now requires per-eye `reprojection_rotation` + `fov`, computed via the new `alvr_rotation_delta()` helper. This is genuinely new capability: real head-rotation reprojection, which the Cardboard distortion renderer never had on its own. |
| `alvr_get_frame` | Returned `(view_params, buffer)` + timestamp | Returns `(timestamp, buffer)` only. View params/foveation centers for that frame are now fetched separately via `alvr_report_compositor_start(timestamp, &metadata)`, which can return `false` (no match) - the caller then reuses the last known-good metadata (implemented via `CTX.lastFrameMetadata`/`hasFrameMetadata`). |
| Foveation params | Parsed by hand from `alvr_get_settings_json()` | Delivered directly on `AlvrEvent::StreamingStarted` (`enable_foveated_encoding`, `foveated_encoding`) |
| `AlvrStreamConfig` | No upscaling fields | New `enable_upscaling`/`upscaling_edge_*`/`upscale_factor` fields (left disabled here) |
| `AlvrClientCapabilities` | - | New `max_view_width/height`, `prefer_10bit`, `preferred_encoding_gamma`, `prefer_hdr` (left at conservative defaults here) |
| Enum style | Flat C enums (`ALVR_EVENT_STREAMING_STARTED`, `ALVR_LOG_LEVEL_ERROR`) | **Unchanged** - still flat/unscoped despite `AlvrEvent` now being a genuine tagged union in Rust. Confirmed against the untouched `utils.h`, which already uses this convention. |

Validated with a syntax-only compile (`g++ -fsyntax-only -Wall -Wextra`) against hand-written
stub headers for `jni.h`, `android/log.h`, `GLES3/gl3.h`, and `cardboard.h`, since the real
Android NDK/Cardboard SDK aren't available in this environment either. This caught two real bugs
before they'd have hit a real build: enums accidentally declared as scoped `enum class` instead of
plain C enums, and reversed pointer constness on `swapchain_textures`. It cannot substitute for
building against the real NDK + regenerated header, since the stubs are approximations.

## Not done yet (separate, later step)

ARCore integration and the rotational-drift complementary filter. `getPose()` in `alvr_main.cpp`
is explicitly commented as the extension point for that work - it currently returns Cardboard's
gyro-only orientation with position pinned to the origin (3DoF).
