"""Local RTC control helpers: executable queue, timing and approach feedback."""
from collections import deque
import math
import numpy as np


# plan_queue
class PlanQueue:
    def __init__(self, actions):
        self.future = {}
        self.replace(actions, origin=0, current=0, delay=0)

    def prefix(self, current, horizon=50):
        result = np.zeros((horizon, 6), dtype=np.float32)
        valid = 0
        for i in range(horizon):
            if current + i not in self.future:
                break
            result[i] = self.future[current + i]
            valid += 1
        return result, valid

    def replace(self, actions, *, origin, current, delay):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (50, 6) or not np.isfinite(actions).all():
            raise ValueError('Invalid action chunk')
        if current < origin or delay < 0:
            raise ValueError('Invalid plan time indices')
        splice = max(current, origin + delay)
        if splice >= origin + len(actions):
            raise TimeoutError('Returned action chunk has expired')
        if any(i not in self.future for i in range(current, splice)):
            raise TimeoutError('Insufficient old plan to retain committed prefix')
        old_at_splice = self.future.get(splice)
        updated = {i:a for i,a in self.future.items() if current <= i < splice}
        for i in range(splice - origin, len(actions)):
            updated[origin + i] = actions[i].copy()
        self.future = updated
        return dict(splice_step=splice, consumed_steps=current-origin,
            committed_until=origin+delay,
            deadline_overrun=current > origin+delay,
            same_time_replacement_delta=None if old_at_splice is None
                else (actions[splice-origin] - old_at_splice).tolist())

    def take(self, step):
        if step not in self.future:
            raise TimeoutError('Action queue empty; stop rather than replay an old action')
        action = self.future.pop(step)
        self.future = {i:a for i,a in self.future.items() if i > step}
        return action.copy()


# retimed_queue
def retime_path(anchor, knots, max_step, samples):
    anchor = np.asarray(anchor, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    max_step = np.asarray(max_step, dtype=np.float64)
    if anchor.shape != (6,) or knots.ndim != 2 or knots.shape[1] != 6 or not len(knots):
        raise ValueError('Expected a six-axis anchor and nonempty six-axis path')
    if max_step.shape != (6,) or np.any(max_step <= 0) or samples < 1:
        raise ValueError('Invalid retiming limits')
    if not all(np.isfinite(x).all() for x in (anchor, knots, max_step)):
        raise ValueError('Nonfinite path or limits')
    path = np.vstack([anchor, knots])
    # At least one control interval per original segment preserves waiting
    # periods. A single scalar duration slows every axis, including gripper.
    duration = np.maximum(1., np.max(np.abs(np.diff(path, axis=0))/max_step, axis=1))
    boundaries = np.r_[0., np.cumsum(duration)]
    ticks = np.arange(1, samples+1, dtype=np.float64)
    result = np.stack([np.interp(ticks, boundaries, path[:,j]) for j in range(6)], axis=1)
    source_index = np.interp(ticks, boundaries, np.arange(-1, len(knots), dtype=float))
    return result, source_index, dict(
        raw_segments=len(knots), required_control_intervals=float(boundaries[-1]),
        path_slowdown=float(boundaries[-1]/len(knots)),
        original_path_fraction_reached=float((source_index[-1]+1)/len(knots)))
class RetimedPlanQueue(PlanQueue):
    def __init__(self, actions, *, start, max_step, smoothing_window=1):
        if not isinstance(smoothing_window, int) or not 1 <= smoothing_window <= 8:
            raise ValueError('Smoothing window must be an integer in 1..8')
        self.future, self.provenance = {}, {}
        self.raw_future = {}
        self.smoothing_window = smoothing_window
        # A causal common filter makes already committed commands immutable.
        # Its history must be the unfiltered reference, not measured positions
        # or filtered commands, to preserve the speed/acceleration bounds.
        self.raw_history = {i: np.asarray(start, dtype=float).copy()
                            for i in range(-smoothing_window, 0)}
        self.max_step = np.asarray(max_step,dtype=float)
        self.last_change = self.replace(actions,origin=0,current=0,delay=0,previous_sent=start)

    def replace(self, actions, *, origin, current, delay, previous_sent):
        actions = np.asarray(actions,dtype=np.float64)
        if actions.shape != (50,6) or not np.isfinite(actions).all():
            raise ValueError('Invalid model chunk')
        if current < origin or delay < 0:
            raise ValueError('Invalid plan timeline')
        splice = max(current,origin+delay)
        offset = splice-origin
        if offset >= 50:
            raise TimeoutError('Returned action chunk has expired')
        if any(i not in self.future for i in range(current,splice)):
            raise TimeoutError('Old queue cannot cover the committed prefix')
        raw_prefix = {**self.raw_history, **self.raw_future}
        if splice-1 not in raw_prefix:
            raise TimeoutError('Missing unfiltered trajectory history at splice')
        anchor = raw_prefix[splice-1]
        old = self.future.get(splice)
        scheduled, progress, stats = retime_path(anchor,actions[offset:],self.max_step,50-offset)
        preserved = {i:a for i,a in self.future.items() if current<=i<splice}
        raw_preserved = {i:a for i,a in self.raw_future.items() if current<=i<splice}
        provenance = {i:a for i,a in self.provenance.items() if current<=i<splice}
        for j, target in enumerate(scheduled):
            raw_preserved[splice+j] = target.copy()
        raw_reference = {**self.raw_history, **raw_preserved}
        for j, target in enumerate(scheduled):
            k = splice+j
            history_indices = range(k-self.smoothing_window+1,k+1)
            if any(i not in raw_reference for i in history_indices):
                raise TimeoutError('Missing smoothing history')
            # Identical nonnegative weights for all six axes: speed and
            # calibrated bounds are preserved by convexity. For W samples,
            # |delta(command_increment)| <= 2*max_step/W per control interval.
            preserved[k] = np.mean([raw_reference[i] for i in history_indices], axis=0)
            provenance[k] = dict(model_origin=origin,model_time_index=offset+j,
                source_model_index=float(offset+progress[j]),
                model_time_target=actions[offset+j].tolist(),
                trajectory_retimed=bool(abs(progress[j]-j)>1e-7),
                unsmoothed_target=target.tolist(),
                smoothing_window=self.smoothing_window,
                trajectory_smoothed=bool(np.any(abs(preserved[k]-target)>1e-7)))
        self.future, self.provenance = preserved, provenance
        self.raw_future = raw_preserved
        self.last_change = dict(splice_step=splice,consumed_steps=current-origin,
            committed_until=origin+delay,deadline_overrun=current>origin+delay,
            # Keep raw-vs-old-schedule difference separate from scheduled
            # changes: a smaller retimed jump is not proof of better VJP.
            same_time_replacement_delta=None if old is None else (actions[offset]-old).tolist(),
            scheduled_replacement_delta=None if old is None else (preserved[splice]-old).tolist(),
            smoothing_window=self.smoothing_window,
            smoothing_group_delay_steps=(self.smoothing_window-1)/2,
            retiming=stats,rtc_prefix_source='executable_retimed_queue')
        return self.last_change

    def details(self, step):
        if step not in self.provenance:
            raise TimeoutError('Missing scheduled-action provenance')
        return self.provenance[step].copy()

    def take(self, step):
        action = super().take(step)
        self.raw_history[step] = self.raw_future.pop(step)
        self.raw_history = {i:a for i,a in self.raw_history.items()
                            if step-self.smoothing_window+1 <= i <= step}
        self.provenance = {i:a for i,a in self.provenance.items() if i>step}
        return action


# rtc_schedule
class DelayEstimator:
    def __init__(self, initial_roundtrip_ms, *, minimum=8, maximum=20):
        if not math.isfinite(initial_roundtrip_ms) or initial_roundtrip_ms < 0:
            raise ValueError('Invalid initial latency')
        self.minimum, self.maximum = minimum, maximum
        self.samples = deque(maxlen=20)
        # predict() adds the scheduling margin once; avoid padding twice at
        # startup, when the first object approach is most sensitive to lag.
        self.bootstrap = initial_roundtrip_ms / 40

    def observe(self, consumed_steps):
        if not isinstance(consumed_steps, int) or consumed_steps < 0:
            raise ValueError('Invalid observed delay')
        self.samples.append(consumed_steps)

    def predict(self, valid, *, minimum=None):
        floor=self.minimum if minimum is None else minimum
        if not isinstance(floor,int) or not 1<=floor<=self.maximum:
            raise ValueError('Invalid RTC delay floor')
        if valid < floor:
            raise TimeoutError('Too few executable actions for the RTC delay floor')
        # Consumption already includes dispatch and result polling.
        estimate = np.percentile(self.samples, 90) if self.samples else self.bootstrap
        return min(valid, self.maximum, max(floor, math.ceil(float(estimate)) + 1))


# rtc_preflight
def run_preflight(request, observation, base_actions, *, samples=3):
    actions=np.asarray(base_actions,dtype=np.float32)
    records=[]
    for index in range(samples+1):
        prefix=np.zeros((50,6),dtype=np.float32)
        prefix[:38]=actions[12:]
        incoming={**observation,'rtc':dict(previous_actions=prefix,valid_length=38,inference_delay=20)}
        result,latency=request(incoming,timeout=15.)
        if not result.get('rtc',{}).get('guided'):
            raise RuntimeError('Preflight did not receive Guided RTC actions')
        actions=np.asarray(result['actions'],dtype=np.float32)
        if actions.shape!=(50,6) or not np.isfinite(actions).all():
            raise ValueError('Invalid preflight action response')
        records.append(dict(warmup=index==0,request_roundtrip_ms=latency,
                            server_policy_wall_ms=result.get('policy_wall_ms'),
                            server_timing=result.get('server_timing')))
        print(f'RTC PREFLIGHT {index+1}/{samples+1}: {latency:.1f} ms; no motion enabled.',flush=True)
    latencies=[r['request_roundtrip_ms'] for r in records if not r['warmup']]
    p95=float(np.percentile(latencies,95));maximum=max(latencies)
    return dict(status='passed' if p95<=700 and maximum<=900 else 'failed',
                warmed_p95_ms=p95,warmed_max_ms=maximum,
                required_p95_ms=700,required_max_ms=900,requests=records,
                robot_commands_sent=False,
                note='Latency gate for this 50-step queue and 1-second observation-age limit; not a guarantee against later network spikes.')


# approach_feedback
class ApproachFeedback:
    def __init__(self, *, lookahead=20, hold_steps=8):
        self.lookahead=lookahead
        self.hold_steps=hold_steps
        self.active_until=-1

    def inspect(self, prefix, valid, measured, step):
        values=np.asarray(prefix,dtype=float)
        state=np.asarray(measured,dtype=float)
        if values.shape!=(50,6) or state.shape!=(6,) or not 0<valid<=50:
            raise ValueError('Invalid approach-feedback input')
        if not np.isfinite(values[:valid]).all() or not np.isfinite(state).all():
            raise ValueError('Nonfinite approach-feedback input')
        gripper=values[:min(valid,self.lookahead),5]
        maximum=float(gripper.max())
        peak=np.maximum.accumulate(np.r_[state[5],gripper])[:-1]
        closing=np.flatnonzero((peak>=18)&(peak-gripper>=8))
        opening=np.flatnonzero((gripper>=18)&(gripper-state[5]>=8))
        reason=None
        if len(closing): reason='predicted_gripper_closing'
        elif len(opening): reason='predicted_gripper_opening'
        elif state[5]>=18: reason='gripper_open'
        if reason is not None:
            self.active_until=max(self.active_until,step+self.hold_steps)
        active=step<=self.active_until
        return dict(active=active,reason=reason or ('hysteresis' if active else 'cruise'),
                    predicted_closing_offset=None if not len(closing) else int(closing[0]),
                    predicted_opening_offset=None if not len(opening) else int(opening[0]),
                    measured_gripper=float(state[5]),future_gripper_max=maximum,
                    note='Gripper-transition heuristic, not visual alignment or grasp detection')
