"""Recover an empty SAM tracker using subsequent actual text detections.

No fabricated point prompt, backwards propagation from a future seed, or
cross-restart object identity is introduced. Successful propagation is unchanged.
"""


def propagate_resilient(predictor, session, prompt, count, record, *, prompt_resets_state=False):
    cursor = 0
    offset = 0
    maximum_label = 0
    recoveries = []
    while cursor < count:
        if cursor and not prompt_resets_state:
            predictor.handle_request({'type': 'reset_session', 'session_id': session})
        while cursor < count:
            response = predictor.handle_request({'type': 'add_prompt', 'session_id': session, 'frame_index': cursor, 'text': prompt})
            detected, largest = record(cursor, response['outputs'], offset)
            maximum_label = max(maximum_label, largest)
            if detected:
                break
            cursor += 1
            if cursor < count and not prompt_resets_state:
                predictor.handle_request({'type': 'reset_session', 'session_id': session})
        if cursor == count:
            break
        seed = cursor
        next_frame = seed + 1
        try:
            for response in predictor.handle_stream_request({'type': 'propagate_in_video', 'session_id': session, 'start_frame_index': seed, 'propagation_direction': 'forward'}):
                index = int(response['frame_index'])
                if not seed <= index < count:
                    raise ValueError('SAM propagation returned a frame outside its committed interval')
                _, largest = record(index, response['outputs'], offset)
                maximum_label = max(maximum_label, largest)
                next_frame = max(next_frame, index + 1)
        except RuntimeError as error:
            if 'No points are provided; please add points first' not in str(error):
                raise
            recoveries.append({'resume_frame': next_frame, 'reason': 'No active tracker points; search subsequent frames for observed text detections'})
            cursor = next_frame
            offset = maximum_label
            continue
        if next_frame != count:
            raise ValueError('SAM propagation ended without covering the complete interval')
        cursor = count
    return recoveries
