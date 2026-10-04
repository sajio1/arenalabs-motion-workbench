"""Use the official recording-clock repair before selecting source rows.

Only timestamp metadata is repaired. Pose samples are neither reordered nor
interpolated, and the original xdata archive is never changed.
"""
import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).with_name('official')/'nymeriaplus/loaders/xsens.py'
spec=importlib.util.spec_from_file_location('_nymeria_official_xsens',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def recording_clock(timestamps):
    raw=np.asarray(timestamps)
    data={'timestamps_us':raw.copy()}
    reader=module.XSensBodyLoader.__new__(module.XSensBodyLoader)
    # Invoke the upstream implementation, including its 10 ms end-drift gate.
    reader._correct_timestamps(data)
    corrected=data['timestamps_us']
    if np.any(np.diff(corrected)<=0):
        raise ValueError('Official XSens correction did not yield a monotonic clock')
    invalid=np.abs(np.diff(raw)-reader._DT_NOMINAL_US)>reader._DT_TOL_US
    report={'policy':'official-XSensBodyLoader-clock-correction',
            'invalid_timestamp_intervals':int(invalid.sum()),
            'end_drift_us':int(corrected[-1]-raw[-1]),
            'space_samples_modified':False}
    return corrected,report

def nearest_source_rows(clock,query):
    found=np.clip(np.searchsorted(clock,query),0,len(clock)-1)
    before=np.maximum(found-1,0)
    return np.where(np.abs(clock[before]-query)<=np.abs(clock[found]-query),before,found)
