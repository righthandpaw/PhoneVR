#!/usr/bin/env bash
# =============================================================================
#  dev.sh — PhoneVR day-to-day helper
#  Usage: ./dev.sh [command]   or just ./dev.sh for interactive menu
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()   { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()     { echo -e "${GREEN}[ OK ]${NC}  $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC}  $1"; }
die()    { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }
header() { echo -e "\n${BOLD}${CYAN}── $1 ──${NC}"; }

ANDROID_DIR="$ANDROID_PROJECT"
APK_DIR="$ANDROID_DIR/app/build/outputs/apk"

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_build() {
    header "Build APK (noGvr Release)"
    cd "$ANDROID_DIR"
    chmod +x ./gradlew
    env PATH="$PATH" \
        ANDROID_HOME="$ANDROID_HOME" \
        ANDROID_SDK_ROOT="$ANDROID_HOME" \
        ANDROID_NDK_HOME="$ANDROID_NDK_HOME" \
        JAVA_HOME="$JAVA_HOME" \
        ./gradlew assembleNoGvrRelease --no-daemon -x lint
    ok "Build complete."
    find "$APK_DIR" -name "*.apk" | while read -r f; do
        echo "  APK: $f"
    done
}

cmd_build_debug() {
    header "Build APK (noGvr Debug — extra logging)"
    cd "$ANDROID_DIR"
    chmod +x ./gradlew
    env PATH="$PATH" \
        ANDROID_HOME="$ANDROID_HOME" \
        ANDROID_SDK_ROOT="$ANDROID_HOME" \
        ANDROID_NDK_HOME="$ANDROID_NDK_HOME" \
        JAVA_HOME="$JAVA_HOME" \
        ./gradlew assembleNoGvrDebug --no-daemon
    ok "Debug build complete."
    find "$APK_DIR" -name "*.apk" | while read -r f; do
        echo "  APK: $f"
    done
}

cmd_install() {
    header "Install APK on connected Android phone"
    # Check a device is connected
    DEVICE=$(adb devices 2>/dev/null | grep -v "List of" | grep "device$" | awk '{print $1}' | head -1)
    if [[ -z "$DEVICE" ]]; then
        die "No device found. Make sure:
  1. USB cable is connected
  2. USB Debugging is enabled (Settings → Developer Options → USB Debugging)
  3. You've accepted the 'Allow USB Debugging' prompt on your phone"
    fi
    ok "Device: $DEVICE ($(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r'))"

    APK=$(find "$APK_DIR/noGvr/release" -name "*.apk" 2>/dev/null | head -1)
    [[ -z "$APK" ]] && die "No APK found. Run build first."
    info "Installing: $(basename "$APK")"
    adb install -r "$APK" && ok "Installed successfully."
}

cmd_logcat() {
    header "Live Phone Logs — PhoneVR & ALVR only (Ctrl+C to stop)"
    warn "Make sure the app is running on your phone first."
    echo ""
    adb logcat -s "PhoneVR" "ALVR" "AndroidRuntime" "System.err"
}

cmd_clean() {
    header "Clean build artifacts"
    cd "$ANDROID_DIR"
    chmod +x ./gradlew
    ./gradlew clean --no-daemon
    ok "Cleaned."
}

cmd_sync_upstream() {
    header "Sync your fork with the original PhoneVR repo"
    cd "$PHONEVR_REPO"
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    info "Fetching from upstream (original PhoneVR)..."
    git fetch upstream
    info "Merging upstream/master into '$BRANCH' ..."
    git merge upstream/master --no-edit || {
        warn "Merge conflict! Resolve the conflicts in the files listed above."
        warn "Then run: git add . && git commit"
        exit 1
    }
    ok "Sync complete."
    info "Push your synced branch with: git push origin $BRANCH"
}

cmd_push() {
    header "Commit & push to your GitHub fork"
    cd "$PHONEVR_REPO"
    BRANCH=$(git rev-parse --abbrev-ref HEAD)

    git add -A
    STAGED=$(git diff --cached --name-only)
    if [[ -z "$STAGED" ]]; then
        warn "Nothing to commit."
        return
    fi

    echo "  Files to commit:"
    echo "$STAGED" | sed 's/^/    /'
    echo ""
    read -rp "  Commit message: " MSG
    [[ -z "$MSG" ]] && MSG="Update"
    git commit -m "$MSG"
    git push origin "$BRANCH"
    ok "Pushed. CI build starting on GitHub automatically."

    REPO_URL=$(gh repo view --json url -q .url 2>/dev/null || echo "")
    [[ -n "$REPO_URL" ]] && echo "  Actions: $REPO_URL/actions"
}

cmd_phone() {
    header "Connected Android device info"
    adb devices -l
    echo ""
    DEVICE=$(adb devices 2>/dev/null | grep -v "List of" | grep "device$" | awk '{print $1}' | head -1)
    if [[ -n "$DEVICE" ]]; then
        ok "Device:  $(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r')"
        info "Android: $(adb shell getprop ro.build.version.release 2>/dev/null | tr -d '\r')"
        info "API:     $(adb shell getprop ro.build.version.sdk 2>/dev/null | tr -d '\r')"
    else
        warn "No device detected. Enable USB Debugging on your phone."
        echo "  Settings → About Phone → tap Build Number 7 times"
        echo "  Settings → Developer Options → USB Debugging → Enable"
    fi
}

cmd_status() {
    header "Git Status"
    cd "$PHONEVR_REPO"
    git status
    echo ""
    git log --oneline -5 --graph --decorate
}

cmd_update_submodules() {
    header "Update git submodules"
    cd "$PHONEVR_REPO"
    git submodule update --init --recursive
    ok "Submodules up to date."
    [[ -f "$ANDROID_DIR/ALVR/Cargo.toml" ]] \
        && ok "ALVR submodule verified." \
        || warn "ALVR/Cargo.toml still missing — something is wrong."
}

# ── Menu ──────────────────────────────────────────────────────────────────────
show_menu() {
    echo -e "\n${BOLD}${CYAN}PhoneVR Dev Helper${NC}"
    echo -e "${CYAN}──────────────────────────────────${NC}"
    echo "  1) Build APK (release)"
    echo "  2) Build APK (debug — more logs)"
    echo "  3) Install APK on connected phone"
    echo "  4) View live phone logs (logcat)"
    echo "  5) Clean build artifacts"
    echo "  6) Sync with original PhoneVR repo"
    echo "  7) Commit & push to GitHub"
    echo "  8) Check connected phone"
    echo "  9) Git status"
    echo "  10) Update submodules"
    echo "  0) Exit"
    echo ""
    read -rp "  Choose: " OPT
    case "$OPT" in
        1)  cmd_build ;;
        2)  cmd_build_debug ;;
        3)  cmd_install ;;
        4)  cmd_logcat ;;
        5)  cmd_clean ;;
        6)  cmd_sync_upstream ;;
        7)  cmd_push ;;
        8)  cmd_phone ;;
        9)  cmd_status ;;
        10) cmd_update_submodules ;;
        0)  exit 0 ;;
        *)  warn "Unknown option." ;;
    esac
}

# Direct command or interactive menu
if [[ $# -gt 0 ]]; then
    case "$1" in
        build)      cmd_build ;;
        debug)      cmd_build_debug ;;
        install)    cmd_install ;;
        logcat)     cmd_logcat ;;
        clean)      cmd_clean ;;
        sync)       cmd_sync_upstream ;;
        push)       cmd_push ;;
        phone)      cmd_phone ;;
        status)     cmd_status ;;
        submodules) cmd_update_submodules ;;
        *) echo "Commands: build debug install logcat clean sync push phone status submodules" ;;
    esac
else
    while true; do show_menu; done
fi
