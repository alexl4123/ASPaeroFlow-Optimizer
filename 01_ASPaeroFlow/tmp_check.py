import pandas as pd
import numpy as np

# --- 1. load the two schedules ------------------------------------------------
# remove "header=None" if your files already have a header row

import sys

if len(sys.argv) != 3:
    print("Usage: python tmp_check.py <initial_file> <final_file>")
    sys.exit(1)

initial_file = sys.argv[1]
final_file = sys.argv[2]

init  = pd.read_csv(initial_file, header=None).to_numpy()  # shape (|I|, |T|)
final = pd.read_csv(final_file, header=None).to_numpy()   # shape (|I|, |T'|)

# --- 2. helper: last non--1 position for every row, fully vectorised ----------
def last_valid_pos(arr: np.ndarray) -> np.ndarray:
    """
    Return a 1-D array with, for every row in `arr`, the **last** column index
    whose value is not -1.  If a row is all -1, we return -1 for that flight.
    """
    # True where value ≠ -1
    mask = arr != -1                                        # same shape as arr

    # Reverse columns so that the *first* True along axis=1 is really the last
    # in the original orientation
    reversed_first = np.argmax(mask[:, ::-1], axis=1)

    # If the whole row was False, argmax returns 0.  Detect that case:
    no_valid = ~mask.any(axis=1)                            # shape (|I|,)

    # Convert “position in reversed array” back to real column index
    last_pos = arr.shape[1] - 1 - reversed_first            # shape (|I|,)
    last_pos[no_valid] = -1                                 # sentinel value

    return last_pos.astype(np.int64)

t_init  = last_valid_pos(init)      # last non--1 in the *initial* schedule
t_final = last_valid_pos(final)     # last non--1 in the *final* schedule

# --- 3. compute delays --------------------------------------------------------
# Flights that disappear completely (-1 in *both* files) get a delay of 0
delay = np.where(t_init >= 0, t_final - t_init, 0)          # shape (|I|,)

# --- 4. aggregate in whichever way you need -----------------------------------
total_delay  = delay.sum()
mean_delay   = delay.mean()
max_delay    = delay.max()
per_flight   = delay.tolist()        # Python list if you want it

print(f"Total delay (all flights): {total_delay}")
print(f"Average delay per flight:  {mean_delay:.2f}")
print(f"Maximum single-flight delay: {max_delay}")
