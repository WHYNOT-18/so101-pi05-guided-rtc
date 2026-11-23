"""Pure limits for the first bounded SO-101 motion trial."""
import numpy as np


def limit_target(predicted, measured, previous, lower, upper, max_step):
    arrays = [np.asarray(x, dtype=float) for x in
              (predicted, measured, previous, lower, upper, max_step)]
    if any(x.shape != (6,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Expected finite six-dimensional motion values')
    predicted, measured, previous, lower, upper, max_step = arrays
    if np.any(lower > upper) or np.any(max_step <= 0):
        raise ValueError('Invalid motion limits')
    if np.any(measured < lower - 0.5) or np.any(measured > upper + 0.5):
        raise RuntimeError('Measured position is outside trial bounds; stop')
    if np.any(np.abs(measured - previous) > np.array([10.] * 5 + [15.])):
        raise RuntimeError('Joint tracking error exceeds trial threshold; stop')
    lo = np.maximum.reduce([lower, measured - max_step, previous - max_step])
    hi = np.minimum.reduce([upper, measured + max_step, previous + max_step])
    if np.any(lo > hi):
        raise RuntimeError('No feasible next target; stop')
    return np.clip(predicted, lo, hi)


def normal_target(predicted, measured, previous, lower, upper, *, joint_step=10., gripper_step=12.):
    """Separate actual tracking failure from a new plan's proposed jump.

    Never increase a failed tracking guard to accommodate a proposed target.
    Rate and relative-target bounds constrain the command instead; these are
    actuator limits, not RTC guidance or a change to the model prediction.
    """
    arrays = [np.asarray(x, dtype=float) for x in (predicted, measured, previous, lower, upper)]
    if any(a.shape != (6,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite or malformed joint target/state')
    predicted, measured, previous, lower, upper = arrays
    if np.any(lower > upper):
        raise ValueError('Invalid calibrated bounds')
    if np.any(measured < lower - .5) or np.any(measured > upper + .5):
        raise RuntimeError('Measured joint outside calibrated range')
    # Compare measured state to what was actually sent, not the next proposal.
    if np.any(abs(previous - measured) > np.array([30.] * 5 + [40.])):
        raise RuntimeError('Actual motor tracking error exceeds 30 degrees / 40 gripper units')
    if not np.isfinite([joint_step,gripper_step]).all() or min(joint_step,gripper_step)<=0:
        raise ValueError('Command step limits must be positive and finite')
    rate = np.array([joint_step] * 5 + [gripper_step])
    relative = np.array([25.] * 5 + [30.])
    lo = np.maximum.reduce([lower, previous - rate, measured - relative])
    hi = np.minimum.reduce([upper, previous + rate, measured + relative])
    if np.any(lo > hi):
        raise RuntimeError('No feasible command under joint/tracking/rate limits')
    return np.clip(predicted, lo, hi)


def checked_scheduled_target(scheduled, measured, previous, lower, upper, max_step):
    """Validate an executable schedule without modifying individual axes."""
    vectors=[np.asarray(x,dtype=float) for x in (scheduled,measured,previous,lower,upper,max_step)]
    if any(x.shape!=(6,) or not np.isfinite(x).all() for x in vectors):
        raise ValueError('Malformed scheduled target, state or bounds')
    target,measured,previous,lower,upper,max_step=vectors
    if np.any(max_step<=0) or np.any(lower>upper):
        raise ValueError('Invalid schedule limits')
    if np.any(measured<lower-.5) or np.any(measured>upper+.5):
        raise RuntimeError('Measured joint outside calibrated range')
    if np.any(target<lower-1e-5) or np.any(target>upper+1e-5):
        raise RuntimeError('Scheduled command outside calibrated range')
    if np.any(abs(previous-measured)>np.array([30.]*5+[40.])):
        raise RuntimeError('Actual tracking error exceeds 30 degrees / 40 gripper units')
    if np.any(abs(target-measured)>np.array([25.]*5+[30.])):
        raise RuntimeError('Robot cannot track the executable schedule; stop rather than distort it')
    if np.any(abs(target-previous)>max_step+1e-5):
        raise RuntimeError('Executable queue violated its shared speed limits')
    return target.copy()
