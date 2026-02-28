#!/bin/bash
# ============================================================================
# MitraSETI — Complete Breakthrough Listen Data Download Script
# ============================================================================
# Total: ~66 files, ~32 GB
# Covers 10 distinct target categories with ON/OFF cadence pairs
# Organized for maximum training diversity and publishable analysis
#
# Uses aria2c for fast parallel downloads with resume support.
# Files that already exist in the destination are skipped automatically.
#
# Usage:
#   cd /path/to/MitraSETI
#   chmod +x data/download_all_BL_data.sh
#   ./data/download_all_BL_data.sh
#
# Files download to: mitraseti_artifacts/data/breakthrough_listen_data_files/
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$(cd "$PROJECT_DIR/.." && pwd)/mitraseti_artifacts/data/breakthrough_listen_data_files"
LIST_FILE="$DEST/.aria2_download_list.txt"
SESSION_FILE="$DEST/.aria2_session"
LOG_FILE="$DEST/.aria2_download.log"

mkdir -p "$DEST"

echo "============================================"
echo " Breakthrough Listen Data Downloader"
echo " Destination: $DEST"
echo "============================================"
echo ""

# ── Build the aria2c input file ──────────────────────────────────────────────
# Each entry: URL on one line, then "  out=filename" + "  dir=DEST" on next lines.
# aria2c's --conditional-get and check-integrity skip existing files.

> "$LIST_FILE"

add() {
    local url="$1"
    local filename
    filename=$(basename "$url")
    if [ -f "$DEST/$filename" ]; then
        echo "  [SKIP] $filename (already exists)"
    else
        echo "$url" >> "$LIST_FILE"
        echo "  out=$filename" >> "$LIST_FILE"
        echo "  [QUEUE] $filename"
    fi
}

# ============================================================================
# CATEGORY 1: BARNARD'S STAR (GJ699) — Nearest single star, 6 ly
# Second-closest star system. High-priority SETI target.
# ON/OFF cadence pairs from two different epochs & instruments
# ============================================================================
echo ""
echo "=== [1/10] BARNARD'S STAR (GJ699) — 8 files ==="

# Epoch 1 (2016, BLC02-05) — ON/OFF/ON cadence
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_81/holding/spliced_blc02030405_2bit_guppi_57457_48006_GJ699_0003.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_81/holding/spliced_blc02030405_2bit_guppi_57457_48334_GJ699_OFF_0004.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_81/holding/spliced_blc02030405_2bit_guppi_57457_48661_GJ699_0005.gpuspec.0002.h5"

# Epoch 2 (2016, BLC10-17) — different instrument/band
add "https://bldata.berkeley.edu/pipeline/AGBT16B_999_55/holding/spliced_blc1011121314151617_guppi_57689_78091_GJ699_0039.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16B_999_55/holding/spliced_blc1011121314151617_guppi_57689_78789_GJ699_0041.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16B_999_55/holding/spliced_blc1011121314151617_guppi_57689_79487_GJ699_0043.gpuspec.0002.h5"

# Medium-resolution (richer spectra)
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_81/holding/spliced_blc02030405_2bit_guppi_57457_48006_GJ699_0003.gpuspec.8.0001.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_81/holding/spliced_blc02030405_2bit_guppi_57457_48334_GJ699_OFF_0004.gpuspec.8.0001.h5"


# ============================================================================
# CATEGORY 2: LALANDE 21185 (GJ411) — 4th closest star, 8.3 ly
# Red dwarf, confirmed exoplanet (GJ 411 b). Two epochs.
# ============================================================================
echo ""
echo "=== [2/10] LALANDE 21185 (GJ411) — 8 files ==="

# Epoch 1 (2016, BLC00-07)
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_213/holding/spliced_blc0001020304050607_guppi_57542_84369_GJ411_0003.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_213/holding/spliced_blc0001020304050607_guppi_57542_85092_GJ411_0005.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_213/holding/spliced_blc0001020304050607_guppi_57542_85812_GJ411_0007.gpuspec.0002.h5"

# Epoch 2 (2016, different date)
add "https://bldata.berkeley.edu/pipeline/AGBT16B_999_32/holding/spliced_blc0001020304050607_guppi_57660_50750_GJ411_0003.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16B_999_32/holding/spliced_blc0001020304050607_guppi_57660_51422_GJ411_0005.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16B_999_32/holding/spliced_blc0001020304050607_guppi_57660_52113_GJ411_0007.gpuspec.0002.h5"

# Medium-resolution pair
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_213/holding/spliced_blc0001020304050607_guppi_57542_84369_GJ411_0003.gpuspec.8.0001.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_213/holding/spliced_blc0001020304050607_guppi_57542_85092_GJ411_0005.gpuspec.8.0001.h5"


# ============================================================================
# CATEGORY 3: TABBY'S STAR (KIC 8462852) — Famous anomalous dimming
# One of the most studied SETI targets. Multiple epochs, ON/OFF cadence.
# ============================================================================
echo ""
echo "=== [3/10] TABBY'S STAR (KIC 8462852) — 8 ON/OFF files ==="

# Epoch 1 (2017 May, BLC20-27) — full ON/OFF/ON/OFF/ON/OFF cadence
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_67/holding/spliced_blc2021222324252627_guppi_57895_36299_DIAG_KIC8462852_0002.gpuspec.0002.h5"

add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_68/holding/spliced_blc2021222324252627_guppi_57898_22410_DIAG_KIC8462852_0031.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_68/holding/spliced_blc2021222324252627_guppi_57898_22733_DIAG_KIC8462852_OFF_0032.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_68/holding/spliced_blc2021222324252627_guppi_57898_23056_DIAG_KIC8462852_0033.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_68/holding/spliced_blc2021222324252627_guppi_57898_23379_DIAG_KIC8462852_OFF_0034.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_68/holding/spliced_blc2021222324252627_guppi_57898_23702_DIAG_KIC8462852_0035.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_68/holding/spliced_blc2021222324252627_guppi_57898_24026_DIAG_KIC8462852_OFF_0036.gpuspec.0002.h5"

# Epoch 2 (2018, BLC10-17 — different instrument configuration)
add "https://bldata.berkeley.edu/pipeline/AGBT18A_999_30/holding/spliced_blc1011121314151617_guppi_58202_42383_KIC8462852_0009.gpuspec.0002.h5"


# ============================================================================
# CATEGORY 4: LHS 292 — Nearby M-dwarf, 14.8 ly
# Faint red dwarf in the solar neighborhood. Two epochs.
# ============================================================================
echo ""
echo "=== [4/10] LHS 292 — 6 ON/OFF files ==="

# Epoch 1 (2016, BLC02-05)
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_157/holding/spliced_blc02030405_2bit_guppi_57500_15126_LHS292_0027.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_157/holding/spliced_blc02030405_2bit_guppi_57500_15452_LHS292_OFF_0028.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_157/holding/spliced_blc02030405_2bit_guppi_57500_15778_LHS292_0029.gpuspec.0002.h5"

# Epoch 2 (2017, BLC00-07)
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_36/holding/spliced_blc0001020304050607_guppi_57842_09544_LHS292_0038.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_36/holding/spliced_blc0001020304050607_guppi_57842_10242_LHS292_0040.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_36/holding/spliced_blc0001020304050607_guppi_57842_10941_LHS292_0042.gpuspec.0002.h5"


# ============================================================================
# CATEGORY 5: EARTH TRANSIT ZONE STAR (HD 109376)
# Star from which Earth would be seen transiting the Sun.
# ============================================================================
echo ""
echo "=== [5/10] EARTH TRANSIT ZONE (HD 109376) — 4 files ==="

add "http://blpd13.ssl.berkeley.edu/ETZ/AGBT17A_999_89/GUPPI/BLP00/blc00_guppi_57934_78131_HD_109376_0054.gpuspec.0002.fil"
add "http://blpd13.ssl.berkeley.edu/ETZ/AGBT17A_999_89/GUPPI/BLP00/blc00_guppi_57934_78728_HD_109376_0056.gpuspec.0002.fil"
add "http://blpd13.ssl.berkeley.edu/ETZ/AGBT17A_999_89/GUPPI/BLP00/blc00_guppi_57934_79326_HD_109376_0058.gpuspec.0002.fil"
add "http://blpd13.ssl.berkeley.edu/ETZ/AGBT17A_999_89/GUPPI/BLP00/blc00_guppi_57934_79922_HD_109376_0060.gpuspec.0002.fil"


# ============================================================================
# CATEGORY 6: HIP STARS — Nearby stars from Hipparcos catalog
# Additional nearby stellar targets for diversity. Multiple systems.
# ============================================================================
echo ""
echo "=== [6/10] HIP STARS — 8 files ==="

# HIP 39826 — ON/OFF cadence
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_26/holding/spliced_blc0001020304050607_guppi_57829_11159_HIP39826_0032.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_26/holding/spliced_blc0001020304050607_guppi_57829_11889_HIP39826_0034.gpuspec.0002.h5"

# HIP 74981
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_191/holding/spliced_blc0001020304050607_guppi_57523_22406_HIP74981_0003.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_191/holding/spliced_blc0001020304050607_guppi_57523_23103_HIP74981_0005.gpuspec.0002.h5"

# HIP 77257
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_48/holding/spliced_blc0001020304050607_guppi_57862_29268_HIP77257_0032.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_48/holding/spliced_blc0001020304050607_guppi_57862_29968_HIP77257_0034.gpuspec.0002.h5"

# HIP 4436
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_08/holding/spliced_blc0001020304050607_guppi_57803_80733_HIP4436_0032.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_08/holding/spliced_blc0001020304050607_guppi_57803_81424_HIP4436_0034.gpuspec.0002.h5"


# ============================================================================
# CATEGORY 7: KEPLER EXOPLANET HOSTS — Confirmed exoplanet systems
# Stars known to host planets from the Kepler mission.
# ============================================================================
echo ""
echo "=== [7/10] KEPLER EXOPLANET HOSTS — 6 files ==="

# KEPLER 1039B — ON/OFF
add "https://bldata.berkeley.edu/pipeline/AGBT18A_999_30/holding/spliced_blc1011121314151617_guppi_58202_44829_kepler1039b_0014.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT18A_999_30/holding/spliced_blc1011121314151617_guppi_58202_45140_kepler1039b_0015.gpuspec.0002.h5"

# KEPLER 738B — ON/OFF
add "https://bldata.berkeley.edu/pipeline/AGBT18A_999_30/holding/spliced_blc1011121314151617_guppi_58202_43360_kepler738b_0011.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT18A_999_30/holding/spliced_blc1011121314151617_guppi_58202_44201_kepler738b_0013.gpuspec.0002.h5"

# KEPLER 992B (higher resolution for diversity)
add "https://bldata.berkeley.edu/pipeline/AGBT18A_999_30/holding/spliced_blc1011121314151617_guppi_58202_43003_kepler992b_0010.gpuspec.0002.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT18A_999_30/holding/spliced_blc1011121314151617_guppi_58202_43003_kepler992b_0010.gpuspec.8.0001.h5"


# ============================================================================
# CATEGORY 8: BARNARD'S STAR + GJ411 — Medium resolution (richer spectra)
# Larger files with more frequency channels for deeper analysis.
# ============================================================================
echo ""
echo "=== [8/10] MEDIUM-RESOLUTION STELLAR FILES — 6 files ==="

add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_81/holding/spliced_blc02030405_2bit_guppi_57457_48661_GJ699_0005.gpuspec.8.0001.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16B_999_55/holding/spliced_blc1011121314151617_guppi_57689_78091_GJ699_0039.gpuspec.8.0001.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_213/holding/spliced_blc0001020304050607_guppi_57542_85812_GJ411_0007.gpuspec.8.0001.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16B_999_32/holding/spliced_blc0001020304050607_guppi_57660_50750_GJ411_0003.gpuspec.8.0001.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_191/holding/spliced_blc0001020304050607_guppi_57523_22406_HIP74981_0003.gpuspec.8.0001.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_68/holding/spliced_blc2021222324252627_guppi_57898_22410_DIAG_KIC8462852_0031.gpuspec.8.0001.h5"


# ============================================================================
# CATEGORY 9: TABBY'S STAR — Medium resolution (complementary)
# Larger bandwidth view of the anomalous star.
# ============================================================================
echo ""
echo "=== [9/10] TABBY'S STAR MEDIUM-RES — 3 files ==="

add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_67/holding/spliced_blc2021222324252627_guppi_57895_36299_DIAG_KIC8462852_0002.gpuspec.8.0001.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_68/holding/spliced_blc2021222324252627_guppi_57898_23056_DIAG_KIC8462852_0033.gpuspec.8.0001.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT18A_999_30/holding/spliced_blc1011121314151617_guppi_58202_42383_KIC8462852_0009.gpuspec.8.0001.h5"


# ============================================================================
# CATEGORY 10: LHS 292 — Medium resolution
# ============================================================================
echo ""
echo "=== [10/10] LHS 292 MEDIUM-RES — 3 files ==="

add "https://bldata.berkeley.edu/pipeline/AGBT16A_999_157/holding/spliced_blc02030405_2bit_guppi_57500_15126_LHS292_0027.gpuspec.8.0001.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_36/holding/spliced_blc0001020304050607_guppi_57842_09544_LHS292_0038.gpuspec.8.0001.h5"
add "https://bldata.berkeley.edu/pipeline/AGBT17A_999_36/holding/spliced_blc0001020304050607_guppi_57842_10242_LHS292_0040.gpuspec.8.0001.h5"


# ── Run aria2c ───────────────────────────────────────────────────────────────

QUEUED=$(grep -c '^http' "$LIST_FILE" 2>/dev/null || echo 0)

echo ""
echo "============================================"

if [ "$QUEUED" -eq 0 ]; then
    echo " All files already downloaded. Nothing to do."
    rm -f "$LIST_FILE"
    echo "============================================"
    exit 0
fi

echo " Queued $QUEUED files for download"
echo " Log:     $LOG_FILE"
echo " Session: $SESSION_FILE"
echo "============================================"
echo ""

aria2c -i "$LIST_FILE" -c \
  -d "$DEST" \
  -x 8 -s 8 -k 1M \
  --retry-wait=5 \
  --max-tries=0 \
  --timeout=60 \
  --connect-timeout=30 \
  --summary-interval=15 \
  --auto-file-renaming=false \
  --file-allocation=none \
  --save-session="$SESSION_FILE" \
  --save-session-interval=30 \
  --log="$LOG_FILE" \
  --console-log-level=notice

echo ""
echo "============================================"
echo " Download complete!"
echo ""
echo " Summary of NEW files by category:"
echo "   Barnard's Star (GJ699):     8 files (~3.5 GB)"
echo "   Lalande 21185 (GJ411):      8 files (~3.5 GB)"
echo "   Tabby's Star (KIC8462852): 11 files (~6.5 GB)"
echo "   LHS 292:                     9 files (~3.5 GB)"
echo "   HD 109376 (Earth TZ):        4 files (~0.3 GB)"
echo "   HIP stars (4 systems):       8 files (~1.9 GB)"
echo "   Kepler hosts (3 systems):    6 files (~2.0 GB)"
echo "   Med-res stellar:             6 files (~5.5 GB)"
echo "   Med-res Tabby's:             3 files (~3.0 GB)"
echo "   Med-res LHS 292:             3 files (~2.5 GB)"
echo "   ---"
echo "   TOTAL NEW:               ~66 files, ~32 GB"
echo ""
echo " Combined with existing 30 files (~14 GB):"
echo "   GRAND TOTAL:             ~96 files, ~46 GB"
echo "   Categories:              10+ distinct target types"
echo "   ON/OFF cadence pairs:    8+ sets"
echo "   Observing epochs:        6+ independent sessions"
echo "   Telescopes:              GBT (100m, Green Bank, WV)"
echo ""
echo " To resume interrupted downloads, re-run this script."
echo "============================================"
