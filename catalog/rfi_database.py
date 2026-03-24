"""
Known RFI Signature Database — MitraSETI

Catalog of known terrestrial radio frequency interference sources
with their characteristic frequencies, bandwidths, and drift rates.
Used to auto-label hits before ML classification, dramatically
reducing review burden.

Sources: ITU Radio Regulations, FCC allocations, satellite databases.

Usage:
    from catalog.rfi_database import RFIDatabase
    rfi_db = RFIDatabase()
    match = rfi_db.match(frequency_mhz=1575.42, drift_rate=0.0)
    if match:
        print(f"Known RFI: {match['source']} ({match['category']})")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RFIEntry:
    source: str
    category: str
    freq_min_mhz: float
    freq_max_mhz: float
    typical_drift_hz_s: float
    notes: str


# Authoritative catalog of known RFI sources in SETI search bands
_RFI_CATALOG: List[RFIEntry] = [
    # GPS constellation
    RFIEntry("GPS L1", "satellite_navigation", 1575.22, 1575.62, 0.0,
             "GPS L1 C/A and P(Y) code"),
    RFIEntry("GPS L2", "satellite_navigation", 1227.40, 1227.80, 0.0,
             "GPS L2 P(Y) and L2C"),
    RFIEntry("GPS L5", "satellite_navigation", 1176.25, 1176.65, 0.0,
             "GPS L5 safety-of-life signal"),
    # GLONASS
    RFIEntry("GLONASS L1", "satellite_navigation", 1598.0625, 1605.375, 0.0,
             "GLONASS L1 FDMA band"),
    RFIEntry("GLONASS L2", "satellite_navigation", 1242.9375, 1248.625, 0.0,
             "GLONASS L2 FDMA band"),
    # Galileo
    RFIEntry("Galileo E1", "satellite_navigation", 1575.22, 1575.62, 0.0,
             "Galileo E1 open service"),
    RFIEntry("Galileo E5a", "satellite_navigation", 1176.25, 1176.65, 0.0,
             "Galileo E5a"),
    RFIEntry("Galileo E5b", "satellite_navigation", 1207.14, 1207.34, 0.0,
             "Galileo E5b"),
    # Iridium
    RFIEntry("Iridium", "satellite_comms", 1616.0, 1626.5, 0.1,
             "Iridium LEO satellite constellation, exhibits drift"),
    # Globalstar
    RFIEntry("Globalstar UL", "satellite_comms", 1610.0, 1618.725, 0.05,
             "Globalstar uplink band"),
    RFIEntry("Globalstar DL", "satellite_comms", 2483.5, 2500.0, 0.05,
             "Globalstar downlink"),
    # Inmarsat
    RFIEntry("Inmarsat L-band", "satellite_comms", 1525.0, 1559.0, 0.0,
             "Inmarsat L-band downlink"),
    # Radio astronomy protected bands (should NOT be RFI but sometimes are)
    RFIEntry("Hydrogen 21cm", "protected_band", 1400.0, 1427.0, 0.0,
             "ITU-protected HI band — leakage indicates nearby RFI"),
    RFIEntry("OH maser", "protected_band", 1610.6, 1613.8, 0.0,
             "Hydroxyl radical maser lines"),
    # Cellular
    RFIEntry("LTE Band 13 UL", "cellular", 777.0, 787.0, 0.0,
             "Verizon 700 MHz uplink"),
    RFIEntry("LTE Band 13 DL", "cellular", 746.0, 756.0, 0.0,
             "Verizon 700 MHz downlink"),
    RFIEntry("LTE Band 71", "cellular", 617.0, 652.0, 0.0,
             "T-Mobile 600 MHz"),
    # Wi-Fi
    RFIEntry("Wi-Fi 2.4 GHz", "wireless_lan", 2400.0, 2483.5, 0.0,
             "IEEE 802.11b/g/n"),
    RFIEntry("Wi-Fi 5 GHz", "wireless_lan", 5150.0, 5850.0, 0.0,
             "IEEE 802.11a/n/ac"),
    # Radar
    RFIEntry("Airport radar", "radar", 2700.0, 2900.0, 0.0,
             "S-band primary surveillance radar"),
    RFIEntry("Weather radar", "radar", 2700.0, 3000.0, 0.0,
             "NEXRAD S-band weather radar"),
    # Broadcast
    RFIEntry("FM broadcast", "broadcast", 87.5, 108.0, 0.0,
             "FM radio broadcast band"),
    RFIEntry("TV UHF", "broadcast", 470.0, 698.0, 0.0,
             "Digital TV UHF channels"),
    # LEO satellite downlinks (Starlink, OneWeb)
    RFIEntry("Starlink DL", "satellite_comms", 10700.0, 12700.0, 0.2,
             "Starlink Ku-band downlink, notable drift from LEO motion"),
    RFIEntry("OneWeb DL", "satellite_comms", 10700.0, 12700.0, 0.2,
             "OneWeb Ku-band downlink"),
    # Common observatory-local RFI
    RFIEntry("60 Hz harmonics", "local_rfi", 0.0, 0.0, 0.0,
             "Power line harmonics — broadband"),
    RFIEntry("Clock harmonics", "local_rfi", 0.0, 0.0, 0.0,
             "Digital clock harmonics from electronics"),
]


class RFIDatabase:
    """Query engine for known RFI signatures."""

    def __init__(self, extra_entries: Optional[List[Dict[str, Any]]] = None):
        self._catalog = list(_RFI_CATALOG)
        if extra_entries:
            for e in extra_entries:
                self._catalog.append(RFIEntry(**e))

    @property
    def catalog(self) -> List[RFIEntry]:
        return self._catalog

    def match(
        self,
        frequency_mhz: float,
        drift_rate: float = 0.0,
        tolerance_mhz: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        """Check if a signal matches a known RFI source.

        Args:
            frequency_mhz: Signal frequency in MHz.
            drift_rate: Signal drift rate in Hz/s.
            tolerance_mhz: Frequency matching tolerance.

        Returns:
            Dict with source info if matched, None otherwise.
        """
        best_match = None
        best_dist = float("inf")

        for entry in self._catalog:
            if entry.freq_min_mhz == 0 and entry.freq_max_mhz == 0:
                continue

            in_band = (
                (entry.freq_min_mhz - tolerance_mhz)
                <= frequency_mhz
                <= (entry.freq_max_mhz + tolerance_mhz)
            )
            if not in_band:
                continue

            center = (entry.freq_min_mhz + entry.freq_max_mhz) / 2
            dist = abs(frequency_mhz - center)

            if dist < best_dist:
                best_dist = dist
                best_match = entry

        if best_match is None:
            return None

        return {
            "source": best_match.source,
            "category": best_match.category,
            "freq_range_mhz": (best_match.freq_min_mhz, best_match.freq_max_mhz),
            "typical_drift": best_match.typical_drift_hz_s,
            "notes": best_match.notes,
            "match_offset_mhz": round(best_dist, 4),
        }

    def match_batch(
        self,
        candidates: List[Dict[str, Any]],
        freq_key: str = "frequency_hz",
        tolerance_mhz: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Label a list of candidates with known RFI matches.

        Adds 'known_rfi' and 'known_rfi_source' keys to matched candidates.
        """
        for c in candidates:
            freq_mhz = c.get(freq_key, 0) / 1e6
            drift = c.get("drift_rate", 0)
            result = self.match(freq_mhz, drift, tolerance_mhz)
            if result:
                c["known_rfi"] = True
                c["known_rfi_source"] = result["source"]
                c["known_rfi_category"] = result["category"]
            else:
                c["known_rfi"] = False
        return candidates

    def summary(self) -> Dict[str, int]:
        """Category counts in the catalog."""
        cats: Dict[str, int] = {}
        for e in self._catalog:
            cats[e.category] = cats.get(e.category, 0) + 1
        return cats
