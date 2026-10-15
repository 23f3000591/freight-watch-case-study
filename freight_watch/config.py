"""All tunable thresholds in one place. Every value is documented in the README."""

# --- Baselines (fixed by the brief) ---
BASELINE_WEEKS = 8            # trailing window for "vs own history"

# --- Anomaly detection ---
MIN_DEV_HISTORY = 8           # prior weekly deviations needed before a week can be flagged
Z_THRESHOLD = 3.0             # robust z-score a deviation must reach to count as "unusual"
SIGMA_FLOOR_PCT = 1.5         # lower bound on the noise estimate, in percentage points
MATERIAL_OWN_PCT = 5.0        # unusual on own-history lens only counts if rise is at least this
MATERIAL_PEER_PCT = 10.0      # unusual on peer lens only counts if gap to peers is at least this

# --- Note verification ---
NOTE_PROXIMITY_DAYS = 28      # unexplained rows mention in-scope notes within this distance
STEP_NOTE_HORIZON_WEEKS = 8   # a "starting this week" note stops explaining a rise after this many
                              # weeks, because the 8-week baseline has absorbed the step by then
