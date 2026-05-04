
import base64
import numpy as np

def encode_ndarray(arr: np.ndarray) -> dict:
    """
    Serializes a NumPy array into a dictionary payload preserving spatial structure and data type.
    """
    # Forces C-order memory continuity. Crucial for arrays generated via slicing.
    contiguous_arr = np.ascontiguousarray(arr)
    return {
        "__ndarray__": base64.b64encode(contiguous_arr.tobytes()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": arr.shape
    }

def decode_ndarray(payload: dict) -> np.ndarray:
    """
    Reconstructs the precise NumPy array from the payload and ensures mutability.
    """
    arr_bytes = base64.b64decode(payload["__ndarray__"])
    dtype = np.dtype(payload["dtype"])
    
    # .copy() forces allocation of a new, writeable C-buffer
    arr = np.frombuffer(arr_bytes, dtype=dtype).copy()
    
    return arr.reshape(payload["shape"])

