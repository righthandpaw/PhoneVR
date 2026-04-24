#!/usr/bin/env bash
# =============================================================================
#  env.sh — Shared environment bootstrap
#
#  Every script sources this file at the top with:
#    source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
#  This means every script is self-contained and works regardless of which
#  shell the user normally uses (fish, zsh, bash, etc). When our scripts run
#  they always run under bash (the shebang line), so we just need to make sure
#  all the right paths are exported here.
# =============================================================================

# ── Android SDK ───────────────────────────────────────────────────────────────
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$ANDROID_HOME/ndk/25.2.9519653}"

# ── Java ──────────────────────────────────────────────────────────────────────
# Try common locations for Java 17 on Arch/CachyOS
for jdir in \
    "/usr/lib/jvm/java-17-openjdk" \
    "/usr/lib/jvm/java-17-openjdk-amd64" \
    "/usr/lib/jvm/temurin-17"; do
    if [[ -d "$jdir" ]]; then
        export JAVA_HOME="$jdir"
        break
    fi
done

# ── Rust / Cargo ──────────────────────────────────────────────────────────────
# Rust installs to ~/.cargo/bin. We add it directly — no need to source
# the bash-only ~/.cargo/env file that broke fish users.
export PATH="$HOME/.cargo/bin:$PATH"

# ── Android SDK tools ─────────────────────────────────────────────────────────
export PATH="$PATH:$ANDROID_HOME/cmdline-tools/latest/bin"
export PATH="$PATH:$ANDROID_HOME/platform-tools"
export PATH="$PATH:$ANDROID_HOME/build-tools/34.0.0"

# ── Convenience ───────────────────────────────────────────────────────────────
export PHONEVR_REPO="$HOME/PhoneVR"
export ANDROID_PROJECT="$PHONEVR_REPO/code/mobile/android/PhoneVR"
