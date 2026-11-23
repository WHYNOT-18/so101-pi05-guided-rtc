"""25 Hz SO-101 control + background cloud Guided RTC. Live preview, no video recording."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import queue
import shutil
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE/'cloud_upload/so101_rtc'), str(HERE/'cloud_upload/so101_remote'),
               str(HERE.parent/'work/openpi/packages/openpi-client/src')]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--port',default='/dev/ttyACM0')
    p.add_argument('--front',type=int,default=4)
    p.add_argument('--wrist',type=int,default=6)
    p.add_argument('--url',default='ws://127.0.0.1:18000')
    p.add_argument('--guidance',choices=('on','off'),default='on',
                   help='Enable RTC guidance or use the same async executable queue without guidance')
    p.add_argument('--seconds',type=float,default=0,
                   help='Run duration; 0 (default) runs until Ctrl+C')
    p.add_argument('--replan-steps',type=int,default=12)
    p.add_argument('--rtc-min-delay',type=int,default=8,
                   help='Minimum committed RTC prefix in control steps')
    p.add_argument('--rtc-soft-tail',type=int,
                   help='Constrain only this many steps after the committed delay; default keeps server tail')
    p.add_argument('--anticipate-grasp',action='store_true',
                   help='Keep cruising interval; request earlier feedback around anticipated gripper transitions')
    p.add_argument('--smoothing-window',type=int,choices=range(1,9),default=4,
                   help='Common causal filter for all six axes; 1 disables smoothing')
    p.add_argument('--joint-speed',type=float,default=60.,help='Joint command speed limit in degrees/second')
    p.add_argument('--gripper-speed',type=float,default=100.,help='Gripper command speed in LeRobot units/second')
    p.add_argument('--no-viewer',action='store_true')
    p.add_argument('--image-codec',choices=('auto','raw','png','jpeg95'),default='auto')
    args=p.parse_args()
    if not args.execute: p.error('Use --execute to enable real robot motion')
    assert math.isfinite(args.seconds) and args.seconds>=0
    assert 5<=args.replan_steps<=25 and args.front!=args.wrist
    assert 4<=args.rtc_min_delay<=12
    assert args.rtc_soft_tail is None or 1<=args.rtc_soft_tail<=6
    assert math.isfinite(args.joint_speed) and 0<args.joint_speed<=120
    assert math.isfinite(args.gripper_speed) and 0<args.gripper_speed<=250
    import numpy as np
    from websockets.sync.client import connect
    from lerobot.motors import Motor,MotorCalibration,MotorNormMode,SerialMotorsBus
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from openpi_client.image_tools import resize_with_pad
    from remote_common import JOINT_NAMES,percentiles,validate_observation
    from controller_support import RetimedPlanQueue, DelayEstimator, run_preflight, ApproachFeedback
    from image_transport import encode_observation,negotiate_codec
    from motion_guard import checked_scheduled_target
    import wire_numpy

    out=HERE/'checks'/('async_guided_rtc_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    source_root=Path(__file__).resolve().parent
    source_paths=[Path(__file__).resolve(),source_root/'motion_guard.py',
        source_root/'cloud_upload/so101_rtc/controller_support.py',
        source_root/'cloud_upload/so101_rtc/image_transport.py']
    (out/'scripts').mkdir()
    source_hashes={}
    for path in source_paths:
        shutil.copy2(path,out/'scripts'/path.name)
        source_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    names=[n.removesuffix('.pos') for n in JOINT_NAMES]
    cal_file=Path('/home/wangbo/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json')
    cal={n:MotorCalibration(**v) for n,v in json.loads(cal_file.read_text()).items()}
    assert set(cal)==set(names)
    bus=SerialMotorsBus(port=args.port,calibration=cal,motors={n:Motor(i,'sts3215',
        MotorNormMode.RANGE_0_100 if n=='gripper' else MotorNormMode.DEGREES)
        for i,n in enumerate(names,1)})
    stopped=threading.Event()
    jobs,results=queue.Queue(maxsize=1),queue.Queue(maxsize=1)
    status,armed,hold_status='failed',False,None
    preflight_status='not_started'
    failure_reason,failure_context=None,None
    step,replacements,limited,retimed_steps=0,0,0,0
    dropped_responses,smoothed_steps=0,0
    late_guidance_acceptances=0
    approach_requests=0
    timings,periods=[],[]
    command_periods=[]
    step_limits=np.array([args.joint_speed]*5+[args.gripper_speed])/25
    config=dict(mode='async_guided_rtc' if args.guidance=='on' else 'async_without_guidance',
        guided_rtc_requested=args.guidance=='on',control_hz=25,horizon=50,
        controller_version='shared_path_v4_2',
        guided_preflight_before_motion=True,
        source_sha256=source_hashes,
        trajectory_retiming='shared_progress_all_six_dimensions',
        rtc_prefix_source='executable_retimed_queue',per_joint_runtime_clipping=False,
        replan_steps=args.replan_steps,min_committed_prefix_steps=args.rtc_min_delay,
        requested_soft_tail_steps=args.rtc_soft_tail,
        anticipate_grasp=args.anticipate_grasp,approach_replan_steps=8,
        debug_grasp_images=False,max_still_cycles=0,
        feedback_profile='anticipatory_gripper_transition' if args.anticipate_grasp else 'fixed_interval',
        late_response_handling='skip_elapsed_and_retime_unexpired_chunk',
        approach_minimum_delay=5,approach_soft_tail_steps=2,
        approach_lookahead_steps=20,
        delay_estimator='rolling_20_p90_observed_control_steps_plus_one',delay_max_steps=20,
        smoothing_window=args.smoothing_window,
        smoothing_group_delay_ms=(args.smoothing_window-1)*20,
        smoothing_method='same_causal_box_filter_all_six_axes_on_planned_reference',
        max_increment_change=(2*step_limits/args.smoothing_window).tolist(),
        initial_pose_envelope=False,artificial_small_step_clipping=False,
        bounds='recorded motor calibration; gripper 0..100',
        tracking_abort_degrees=30,tracking_abort_gripper_units=40,
        command_speed_degrees_per_second=args.joint_speed,
        command_speed_gripper_units_per_second=args.gripper_speed,
        command_step_limit_degrees=args.joint_speed/25,
        command_step_limit_gripper_units=args.gripper_speed/25,
        command_relative_to_measured_degrees=25,command_relative_to_measured_gripper=30,
        seconds=args.seconds,unlimited_duration=args.seconds==0,video_recorded=False)
    config.update(port=args.port,front_camera=args.front,wrist_camera=args.wrist,url=args.url,
                  websocket_proxy_enabled=False)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(f'ASYNC: guidance={args.guidance.upper()}, actual motion, 25 Hz, no video recording.',flush=True)
    print('SHARED PATH RETIMING: arm and gripper share one path clock; RTC prefix is executable.',flush=True)
    print(f'V4.2: replan={args.replan_steps}, smoothing={args.smoothing_window} samples '
          f'({(args.smoothing_window-1)*20:g} ms constant-velocity phase delay).',flush=True)
    print(f'RTC REACTIVITY: minimum_delay={args.rtc_min_delay}, '
          f'soft_tail={args.rtc_soft_tail if args.rtc_soft_tail is not None else "server_default"}.',flush=True)
    if args.anticipate_grasp:
        print(f'APPROACH FEEDBACK: cruise={args.replan_steps}, approach<=8, lookahead=20, soft_tail=2.',flush=True)
    print('Run until Ctrl+C.' if args.seconds==0 else f'Run for {args.seconds:g} seconds.',flush=True)
    print('REPORT DIRECTORY:',out,flush=True)
    try:
        with ExitStack() as cleanup:
            ws=cleanup.enter_context(connect(args.url,proxy=None,compression=None,max_size=4*1024*1024,
                open_timeout=10,ping_interval=None,close_timeout=2))
            message=ws.recv(timeout=10)
            if isinstance(message,str): raise RuntimeError(message)
            metadata=wire_numpy.unpackb(message)
            (out/'server_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            image_codec=negotiate_codec(metadata,args.image_codec)
            config.update(image_codec=image_codec)
            (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            print('IMAGE TRANSPORT:',image_codec,flush=True)
            if args.guidance=='on':
                assert metadata.get('guided_rtc_enabled') and metadata.get('rtc_method')=='VJP_soft_inpainting', \
                    'Cloud service has no Guided RTC; start serve_guided_rtc.py first'
                assert metadata.get('rtc_beta')==5 and metadata.get('rtc_soft_prefix_steps')==6, \
                    'Expected cloud RTC beta=5 and soft-prefix-steps=6; restart the reactive service'
            assert metadata['joint_names']==JOINT_NAMES and metadata['action_shape']==[50,6]
            assert metadata['action_mode']=='absolute_joint_targets'
            norm=HERE/'openpi_assets/pi05_so101_data/so101/norm_stats.json'
            assert metadata['norm_stats_sha256']==hashlib.sha256(norm.read_bytes()).hexdigest()
            print('CLOUD RTC CONFIG:',json.dumps(dict(beta=metadata.get('rtc_beta'),
                soft_prefix_steps=metadata.get('rtc_soft_prefix_steps','full_remaining_prefix'))),flush=True)
            (out/'server_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            cleanup.callback(lambda:bus.disconnect(disable_torque=False) if bus.is_connected else None)
            bus.connect()
            assert bus.is_calibrated,'Follower calibration mismatch'
            modes=bus.sync_read('Operating_Mode',normalize=False,num_retry=2)
            assert all(int(modes[n])==bus._definition(n).operating_modes['position'] for n in names)
            low=bus._normalize({cal[n].id:cal[n].range_min for n in names})
            high=bus._normalize({cal[n].id:cal[n].range_max for n in names})
            a=np.array([low[cal[n].id] for n in names]);b=np.array([high[cal[n].id] for n in names])
            lower,upper=np.minimum(a,b),np.maximum(a,b)
            config.update(lower_bounds=lower.tolist(),upper_bounds=upper.tolist())
            (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            cameras={}
            for name,index in [('front',args.front),('wrist',args.wrist)]:
                cam=OpenCVCamera(OpenCVCameraConfig(index_or_path=index,width=640,height=480,fps=30,fourcc='MJPG'))
                cleanup.callback(lambda c=cam:c.disconnect() if c.is_connected else None)
                cam.connect();cameras[name]=cam
            def read_state():
                raw=bus.sync_read('Present_Position',num_retry=2)
                state=np.array([raw[n] for n in names],dtype=np.float32)
                if not np.isfinite(state).all(): raise ValueError('Nonfinite robot state')
                return state
            prompt='Pick up the red block and place it in the box, then pick up the blue block and place it in the box.'
            def observation(state):
                images,timestamps={},{}
                for name,cam in cameras.items():
                    with cam.frame_lock:
                        if cam.latest_frame is None or cam.latest_timestamp is None:
                            raise RuntimeError('Camera has no frame')
                        images[name]=cam.latest_frame.copy();timestamps[name]=cam.latest_timestamp
                    if time.perf_counter()-timestamps[name]>.15: raise TimeoutError(name+': stale image')
                return validate_observation(dict(front=resize_with_pad(images['front'],224,224),
                    wrist=resize_with_pad(images['wrist'],224,224),state=state.copy(),prompt=prompt)),timestamps
            def bounded(actions):
                actions=np.asarray(actions,dtype=np.float32)
                if actions.shape!=(50,6) or not np.isfinite(actions).all():
                    raise ValueError('Invalid cloud action chunk')
                return np.clip(actions,lower,upper).astype(np.float32)
            def request(obs,timeout=2.):
                tick=time.perf_counter()
                ws.send(wire_numpy.packb(encode_observation(obs,image_codec)))
                raw=ws.recv(timeout=timeout)
                if isinstance(raw,str): raise RuntimeError(raw)
                result=wire_numpy.unpackb(raw)
                bounded(result['actions'])
                return result,(time.perf_counter()-tick)*1000

            rr=None
            if not args.no_viewer:
                import rerun as rr
                import rerun_cli
                rr.init('SO101 asynchronous Guided RTC',spawn=False)
                rr.spawn(executable_path=str(Path(rerun_cli.__file__).resolve().parent/'rerun'),memory_limit='1GiB')
                def preview():
                    try:
                        while not stopped.is_set():
                            for name,cam in cameras.items():
                                with cam.frame_lock:
                                    image=None if cam.latest_frame is None else cam.latest_frame.copy()
                                if image is not None: rr.log('cameras/'+name,rr.Image(image))
                            stopped.wait(.1)
                    except Exception as error:
                        print('Viewer stopped:',error,flush=True)
                viewer=threading.Thread(target=preview,daemon=True);viewer.start()
                cleanup.callback(lambda:(stopped.set(),viewer.join(timeout=1)))

            # Ordinary first chunk is obtained before enabling motion.
            initial_state=read_state()
            first_state_time=time.perf_counter()
            first_obs,first_camera_times=observation(initial_state)
            first,latency=request(first_obs,timeout=15.)
            initial_actions=bounded(first['actions'])
            if args.guidance=='on':
                preflight=run_preflight(request,first_obs,initial_actions)
                preflight_status=preflight['status']
                (out/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
                if preflight['status']!='passed':
                    raise TimeoutError(f'RTC link too slow before motion: P95={preflight["warmed_p95_ms"]:.0f} ms, '
                        f'max={preflight["warmed_max_ms"]:.0f} ms; check the compressed SSH tunnel/network')
                # Refresh the scene after the stationary link test. Its
                # synthetic preflight prefix must never drive the arm.
                initial_state=read_state()
                first_state_time=time.perf_counter()
                first_obs,first_camera_times=observation(initial_state)
                first,latency=request(first_obs,timeout=15.)
                initial_actions=bounded(first['actions'])
            timings.append(latency)
            delay_estimator=DelayEstimator(preflight['warmed_p95_ms'] if args.guidance=='on' else latency,
                                           minimum=args.rtc_min_delay)
            def infer_worker():
                while not stopped.is_set():
                    try: job=jobs.get(timeout=.1)
                    except queue.Empty: continue
                    try:
                        result,latency=request(job['observation'])
                        if args.guidance=='on' and not result.get('rtc',{}).get('guided'):
                            raise RuntimeError('Server did not apply Guided RTC to a nonempty prefix')
                        if args.guidance=='off' and result.get('rtc',{}).get('guided'):
                            raise RuntimeError('Unguided comparison unexpectedly received a guided prediction')
                        results.put_nowait(dict(job=job,result=result,latency=latency))
                    except Exception as error:
                        try: results.put_nowait(dict(error=repr(error)))
                        except queue.Full: pass
                        return
            worker=threading.Thread(target=infer_worker,daemon=True)
            def stop_worker():
                stopped.set();ws.close()
                if worker.is_alive(): worker.join(timeout=1)
            cleanup.callback(stop_worker)
            def halt():
                nonlocal hold_status
                if armed and bus.is_connected:
                    try:
                        raw=bus.sync_read('Present_Position',normalize=False,num_retry=2)
                        bus.sync_write('Goal_Position',raw,normalize=False)
                        hold_status='measured_position_hold_sent'
                    except Exception as error:
                        hold_status='hold_failed: '+repr(error)
                        print('HOLD FAILED: use physical power cutoff if needed:',error,flush=True)
            cleanup.callback(halt)
            if np.any(abs(read_state()-initial_state)>1):
                raise RuntimeError('Arm moved during initialization; restart from stable pose')
            first_target=initial_actions[0]
            if np.any(abs(first_target-initial_state)>np.array([15.]*5+[20.])):
                raise RuntimeError('First predicted target too far from starting pose; inspect scene/pose')
            raw_hold=bus.sync_read('Present_Position',normalize=False,num_retry=2)
            bus.sync_write('Goal_Position',raw_hold,normalize=False)
            armed=True;bus.enable_torque(num_retry=2)
            previous_sent=read_state()
            plan=RetimedPlanQueue(initial_actions,start=previous_sent,max_step=step_limits,
                                 smoothing_window=args.smoothing_window)
            (out/'initial_retiming.json').write_text(json.dumps(plan.last_change,indent=2)+'\n')
            worker.start()
            print('TORQUE ENABLED; continuous 25 Hz control starts.',flush=True)
            start=next_tick=time.perf_counter()
            previous_command_time=start
            previous_time=None
            previous_increment=np.zeros(6)
            splice_events={}
            approach=ApproachFeedback()
            pending=False;last_requested=0;pending_job=None
            with (out/'control.jsonl').open('w',buffering=1) as control, \
                 (out/'replacements.jsonl').open('w',buffering=1) as changes, \
                 (out/'predictions.jsonl').open('w',buffering=1) as predictions:
                predictions.write(json.dumps(dict(origin=0,actions=initial_actions.tolist(),
                    state=initial_state.tolist(),state_read_completed=first_state_time,
                    camera_host_timestamps=first_camera_times))+'\n')
                while args.seconds==0 or time.perf_counter()-start<args.seconds:
                    time.sleep(max(0,next_tick-time.perf_counter()))
                    tick=time.perf_counter()
                    if tick-next_tick>.12: raise TimeoutError('Control thread missed more than 120 ms')
                    if previous_time is not None: periods.append((tick-previous_time)*1000)
                    previous_time=tick
                    measured=read_state()
                    state_read_completed=time.perf_counter()
                    if np.any(measured<lower-.5) or np.any(measured>upper+.5):
                        raise RuntimeError('Measured joint outside calibrated range')
                    try: returned=results.get_nowait()
                    except queue.Empty: returned=None
                    if returned is not None:
                        pending=False
                        pending_job=None
                        if 'error' in returned: raise RuntimeError(returned['error'])
                        job,result=returned['job'],returned['result']
                        if 'rtc' in job['observation']:
                            expected=job['observation']['rtc']
                            returned_rtc=result.get('rtc',{})
                            assert returned_rtc.get('inference_delay')==job['delay'], 'Cloud RTC delay mismatch'
                            assert returned_rtc.get('valid_length')==expected['valid_length'], 'Cloud RTC prefix mismatch'
                            if job.get('approach_feedback',{}).get('active'):
                                assert returned_rtc.get('attention_end')==expected['valid_length'], \
                                    'Cloud did not apply the shorter approach-feedback prefix'
                        consumed=step-job['origin']
                        delay_estimator.observe(consumed)
                        accepted_time=time.perf_counter()
                        timings.append(returned['latency'])
                        actions=bounded(result['actions'])
                        rtc=result.get('rtc',{})
                        beyond_guidance=(args.guidance=='on' and consumed>=rtc.get('attention_end',50))
                        observation_age_ms=(accepted_time-job['capture_time'])*1000
                        # Keep every received valid prediction, including any
                        # that the following age/horizon checks reject.
                        predictions.write(json.dumps(dict(origin=job['origin'],actions=actions.tolist(),
                            state=job['observation']['state'].tolist(),rtc=rtc,
                            received_at_step=step,observation_age_ms=observation_age_ms,
                            late_after_soft_guidance=beyond_guidance))+'\n')
                        _,old_queue_steps=plan.prefix(step)
                        failure_context=dict(step=step,request_origin=job['origin'],
                            response_consumed_steps=consumed,remaining_queue_steps=old_queue_steps,
                            remaining_model_steps=max(0,50-consumed),
                            rtc_attention_end=rtc.get('attention_end'),
                            observation_age_ms=observation_age_ms)
                        if accepted_time-job['capture_time']>1.0:
                            raise TimeoutError('RTC response based on observation older than 1 s')
                        if consumed>=50:
                            raise TimeoutError('Returned RTC action chunk has completely expired')
                        # A soft inpainting mask is NOT a chunk expiration
                        # boundary. The model still predicts all 50 steps.
                        # replace() starts at max(current, origin+delay), skips
                        # elapsed indices and re-times the remaining future
                        # from the existing command history. All motion guards
                        # below still apply before sending any new command.
                        change=plan.replace(actions,origin=job['origin'],current=step,delay=job['delay'],
                                            previous_sent=previous_sent)
                        failure_context=None
                        late_guidance_acceptances+=int(beyond_guidance)
                        splice_events={k:v for k,v in splice_events.items() if k<change['splice_step']}
                        splice_events[change['splice_step']]=dict(origin=job['origin'],
                            capture_time=job['capture_time'],accepted_time=accepted_time)
                        replacements+=1
                        change.update(request_origin=job['origin'],accepted_at_step=step,
                            guidance_requested=args.guidance=='on',
                            observation_age_at_accept_ms=(time.perf_counter()-job['capture_time'])*1000,
                            request_roundtrip_ms=returned['latency'],rtc=rtc,
                            state_read_completed=job['state_read_completed'],
                            camera_host_timestamps=job['camera_host_timestamps'],
                            client_capture_time=job['capture_time'],client_accept_time=accepted_time,
                            server_policy_wall_ms=result.get('policy_wall_ms'),
                            server_timing=result.get('server_timing'),
                            executable_prefix_length=job['executable_prefix_length'],
                            guidance_prefix_length=job['guidance_prefix_length'],
                            approach_feedback=job.get('approach_feedback'))
                        change.update(late_after_soft_guidance=beyond_guidance,
                            old_queue_steps_at_response=old_queue_steps,
                            remaining_model_steps=50-(change['splice_step']-job['origin']),
                            latency_compensation='skip_elapsed_and_retime_remaining')
                        changes.write(json.dumps(change)+'\n')
                        print(f'RTC {"late-aligned" if beyond_guidance else "replace"} #{replacements}: '
                              f'step={step}, splice={change.get("splice_step")}, '
                              f'latency={returned["latency"]:.1f}ms',flush=True)
                    prefix,valid=plan.prefix(step)
                    feedback=approach.inspect(prefix,valid,measured,step) if args.anticipate_grasp and valid else dict(active=False,reason='cruise')
                    interval=min(args.replan_steps,8) if feedback['active'] else args.replan_steps
                    if not pending and step-last_requested>=interval:
                        floor=min(args.rtc_min_delay,5) if feedback['active'] else args.rtc_min_delay
                        delay=delay_estimator.predict(valid,minimum=floor)
                        tail=2 if feedback['active'] else args.rtc_soft_tail
                        guidance_valid=valid if tail is None else min(valid,delay+tail)
                        obs,camera_times=observation(measured)
                        if args.guidance=='on':
                            obs['rtc']=dict(previous_actions=prefix,valid_length=guidance_valid,inference_delay=delay)
                        pending_job=dict(observation=obs,origin=step,delay=delay,
                            capture_time=min(tick,*camera_times.values()),
                            state_read_completed=state_read_completed,camera_host_timestamps=camera_times,
                            executable_prefix_length=valid,guidance_prefix_length=guidance_valid,
                            approach_feedback=feedback)
                        jobs.put_nowait(pending_job)
                        approach_requests+=int(feedback['active'])
                        pending=True;last_requested=step
                    if step not in plan.future:
                        failure_context=dict(step=step,pending=pending,
                            request_origin=None if pending_job is None else pending_job['origin'],
                            observation_age_ms=None if pending_job is None else
                                (time.perf_counter()-pending_job['capture_time'])*1000,
                            worker_alive=worker.is_alive(),remaining_queue_steps=0)
                        raise TimeoutError('Action queue exhausted before the cloud RTC response arrived; '
                                           'stopping instead of replaying stale actions')
                    source=plan.details(step)
                    scheduled=plan.take(step)
                    proposed=np.asarray(source['model_time_target'])
                    try:
                        target=checked_scheduled_target(scheduled,measured,previous_sent,
                                                        lower,upper,step_limits)
                    except (ValueError,RuntimeError):
                        failure_context=dict(step=step,measured=measured.tolist(),
                            previous_sent=previous_sent.tolist(),predicted_target=proposed.tolist(),
                            scheduled_target=scheduled.tolist(),
                            target_minus_measured=(scheduled-measured).tolist())
                        raise
                    command_limited=bool(np.any(abs(target-scheduled)>1e-6))
                    limited+=int(command_limited)
                    retimed_steps+=int(source['trajectory_retimed'])
                    smoothed_steps+=int(source.get('trajectory_smoothed',False))
                    actual_command_step=(target-previous_sent).tolist()
                    if np.any(abs(np.asarray(actual_command_step)-previous_increment)>
                              2*step_limits/args.smoothing_window+1e-5):
                        raise RuntimeError('Shared smoothing violated command increment-change bounds')
                    # Do not catch up by issuing several time-indexed commands
                    # back-to-back. Keep the common path clock on real time.
                    if time.perf_counter()-previous_command_time>.16:
                        raise TimeoutError('Control cycle took over 160 ms; stop the queued trajectory')
                    time.sleep(max(0,previous_command_time+.04-time.perf_counter()))
                    bus.sync_write('Goal_Position',{n:float(v) for n,v in zip(names,target)})
                    sent_time=time.perf_counter()
                    command_dt=sent_time-previous_command_time
                    command_periods.append(command_dt*1000)
                    previous_command_time=sent_time
                    splice_event=splice_events.pop(step,None)
                    if splice_event is not None:
                        splice_event.update(actual_send_time=sent_time,
                            observation_age_at_first_send_ms=(sent_time-splice_event['capture_time'])*1000,
                            wait_after_accept_ms=(sent_time-splice_event['accepted_time'])*1000)
                    control.write(json.dumps(dict(step=step,elapsed_s=time.perf_counter()-start,
                        measured=measured.tolist(),proposed=proposed.tolist(),sent=target.tolist(),
                        scheduled=scheduled.tolist(),retiming_source=source,
                        command_limited=command_limited,pending=pending,
                        command_step=actual_command_step,
                        command_limit_dt_s=command_dt,
                        command_increment_change=(np.asarray(actual_command_step)-previous_increment).tolist(),
                        state_read_completed=state_read_completed,command_send_completed=sent_time,
                        rtc_splice_event=splice_event,
                        approach_feedback=feedback,active_request_interval_steps=interval,
                        previous_target_tracking_error=(previous_sent-measured).tolist(),
                        queued_steps=len(plan.future)))+'\n')
                    previous_sent=target.copy()
                    previous_increment=np.asarray(actual_command_step)
                    # Process the next observation after the last send and
                    # schedule its command no earlier than one full interval.
                    # Do not accumulate write overhead against an old deadline.
                    step+=1;next_tick=sent_time
                    if step%25==0: print(f'CONTROL {step} steps; queue={len(plan.future)}; RTC replacements={replacements}',flush=True)
            status='completed'
    except KeyboardInterrupt:
        status='interrupted';print('Stopping new actions.',flush=True)
    except Exception as error:
        failure_reason=repr(error)
        print('ASYNC RTC STOPPED:',repr(error),flush=True)
        raise
    finally:
        stopped.set()
        report=dict(status=status,control_steps=step,rtc_replacements=replacements,
            controller_version='shared_path_v4_2',trajectory_retimed_steps=retimed_steps,
            rtc_min_delay=args.rtc_min_delay,requested_soft_tail_steps=args.rtc_soft_tail,
            anticipate_grasp=args.anticipate_grasp,approach_feedback_requests=approach_requests,
            trajectory_smoothed_steps=smoothed_steps,smoothing_window=args.smoothing_window,
            smoothing_group_delay_ms=(args.smoothing_window-1)*20,
            dropped_responses=dropped_responses,
            late_guidance_acceptances=late_guidance_acceptances,
            grasp_stills=None,
            preflight_status=preflight_status,motor_torque_enabled_by_this_run=armed,
            per_joint_runtime_clipping=False,rtc_prefix_source='executable_retimed_queue',
            limited_command_steps=limited,
            joint_speed_limit=args.joint_speed,gripper_speed_limit=args.gripper_speed,
            replan_steps=args.replan_steps,
            guided_rtc_enabled=args.guidance=='on',video_recorded=False,stop_hold=hold_status,
            torque_disabled_on_exit=False,initial_pose_envelope=False,
            failure_reason=failure_reason,failure_context=failure_context)
        if timings: report['request_roundtrip']=percentiles(timings)
        if periods: report['control_period']=percentiles(periods)
        if command_periods: report['command_period']=percentiles(command_periods)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('REPORT:',out/'report.json',flush=True)


if __name__=='__main__': main()
