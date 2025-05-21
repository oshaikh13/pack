import json

# --- Configuration Thresholds (in seconds) ---
KEY_CLICK_MAX_DELTA = 0.7  # Max time between key press and release for a 'key_click'
MOUSE_CLICK_MAX_DELTA = 0.7 # Max time between mouse press and release for a 'mouse_click'
TYPING_MAX_INTERKEY_DELTA = 1.0  # Max time between character key_clicks to form a 'typed_string'
MOUSE_SEQUENCE_MAX_DELTA = 0.5 # Max time between mouse moves/scrolls to be condensed

# --- Helper Functions ---
def is_char_key(key_repr: str) -> bool:
    """Checks if a key representation is a printable character or space."""
    if not key_repr:
        return False
    if key_repr.startswith("Key."):
        return key_repr == "Key.space" # Count space as a character for typing
    return True # Assume single chars like 'a', 'A', '1', '$' are printable

def to_char(key_repr: str) -> str:
    """Converts Key.space to ' ' for typed strings."""
    if key_repr == "Key.space":
        return " "
    return key_repr

class EventCompressor:
    def __init__(self, output_file_path):
        self.output_file = open(output_file_path, "w", encoding="utf-8")
        self.pending_events = {} # Store pending press events, e.g., {'keyboard_Key.ctrl': event}
        
        # Buffers for sequences
        self.typed_char_buffer = [] # Store dicts {'char': char, 'ts': ts, 'duration': dur}
        self.last_typed_char_ts = 0

        self.mouse_move_buffer = [] # Store move events
        self.last_mouse_move_ts = 0

        self.mouse_scroll_buffer = [] # Store scroll events (raw)
        self.last_mouse_scroll_ts = 0

    def _write_event(self, event_data):
        self.output_file.write(json.dumps(event_data, ensure_ascii=False) + "\n")

    def _flush_typed_char_buffer(self):
        if not self.typed_char_buffer:
            return
        
        first_event = self.typed_char_buffer[0]
        last_event = self.typed_char_buffer[-1]
        
        combined_string = "".join([item['char'] for item in self.typed_char_buffer])
        start_ts = first_event['ts']
        # Duration is from the start of the first char press to the end of the last char release
        end_ts = last_event['ts'] + last_event['duration'] 
        
        self._write_event({
            "ts": start_ts,
            "device": "keyboard",
            "type": "typed_string",
            "string": combined_string,
            "duration": round(end_ts - start_ts, 5),
            "num_chars": len(combined_string)
        })
        self.typed_char_buffer = []
        self.last_typed_char_ts = 0

    def _flush_mouse_move_buffer(self):
        if not self.mouse_move_buffer:
            return
        
        first_move = self.mouse_move_buffer[0]
        last_move = self.mouse_move_buffer[-1]
        
        self._write_event({
            "ts": first_move['ts'],
            "device": "mouse",
            "type": "condensed_move",
            "start_x": first_move['x'],
            "start_y": first_move['y'],
            "end_x": last_move['x'],
            "end_y": last_move['y'],
            "duration": round(last_move['ts'] - first_move['ts'], 5),
            "num_moves": len(self.mouse_move_buffer)
        })
        self.mouse_move_buffer = []
        self.last_mouse_move_ts = 0

    def _flush_mouse_scroll_buffer(self):
        if not self.mouse_scroll_buffer:
            return
        
        first_scroll = self.mouse_scroll_buffer[0]
        last_scroll = self.mouse_scroll_buffer[-1] # ts of the last event
        
        total_dx = sum(s['dx'] for s in self.mouse_scroll_buffer)
        total_dy = sum(s['dy'] for s in self.mouse_scroll_buffer)

        # Optional: Filter out if total_dx and total_dy are zero
        # if total_dx == 0 and total_dy == 0 and len(self.mouse_scroll_buffer) > 1:
        #     # This means a series of 0,0 scrolls. Maybe log one or none.
        #     # For now, we log it.
        #     pass

        self._write_event({
            "ts": first_scroll['ts'],
            "device": "mouse",
            "type": "condensed_scroll",
            "total_dx": total_dx,
            "total_dy": total_dy,
            "duration": round(last_scroll['ts'] - first_scroll['ts'], 5),
            "num_scrolls": len(self.mouse_scroll_buffer)
        })
        self.mouse_scroll_buffer = []
        self.last_mouse_scroll_ts = 0
        
    def _flush_all_buffers(self, current_ts=float('inf')):
        """Flushes buffers if current event is too far in time or different type."""
        if self.typed_char_buffer and (current_ts - self.last_typed_char_ts > TYPING_MAX_INTERKEY_DELTA):
            self._flush_typed_char_buffer()
        if self.mouse_move_buffer and (current_ts - self.last_mouse_move_ts > MOUSE_SEQUENCE_MAX_DELTA):
            self._flush_mouse_move_buffer()
        if self.mouse_scroll_buffer and (current_ts - self.last_mouse_scroll_ts > MOUSE_SEQUENCE_MAX_DELTA):
            self._flush_mouse_scroll_buffer()

    def process_event(self, event):
        ts = event['ts']
        device = event['device']
        event_type = event['type']

        # --- Buffer Flushing Logic based on new event type/time ---
        # If new event is not what's being buffered, or too much time has passed, flush.
        if device != "keyboard" or not is_char_key(event.get("key", "")):
            self._flush_typed_char_buffer()
        if device != "mouse" or event_type != "move":
            self._flush_mouse_move_buffer()
        if device != "mouse" or event_type != "scroll":
            self._flush_mouse_scroll_buffer()
        
        # Flush buffers if current event is too far in time from last buffered event
        self._flush_all_buffers(ts)


        # --- Keyboard Event Processing ---
        if device == "keyboard":
            key = event['key']
            pending_key_id = f"keyboard_{key}"

            if event_type == "press":
                # If there's an unreleased press for this key, log it as is (should not happen often)
                if pending_key_id in self.pending_events:
                    self._write_event(self.pending_events.pop(pending_key_id))
                self.pending_events[pending_key_id] = event
            
            elif event_type == "release":
                if pending_key_id in self.pending_events:
                    press_event = self.pending_events.pop(pending_key_id)
                    duration = round(ts - press_event['ts'], 5)
                    
                    if duration <= KEY_CLICK_MAX_DELTA:
                        # It's a key_click
                        key_click_event = {
                            "ts": press_event['ts'],
                            "device": "keyboard",
                            "type": "key_click",
                            "key": key,
                            "duration": duration
                        }
                        # If it's a character, try to add to typed_string_buffer
                        if is_char_key(key):
                            char_to_add = to_char(key)
                            # Check if this char click is close enough to the last one
                            if not self.typed_char_buffer or \
                               (press_event['ts'] - self.last_typed_char_ts <= TYPING_MAX_INTERKEY_DELTA):
                                self.typed_char_buffer.append({
                                    'char': char_to_add, 
                                    'ts': press_event['ts'], 
                                    'duration': duration # duration of this specific key click
                                })
                                self.last_typed_char_ts = press_event['ts'] # Time of the press
                            else: # Too far, flush old buffer, start new one
                                self._flush_typed_char_buffer()
                                self.typed_char_buffer.append({
                                    'char': char_to_add, 
                                    'ts': press_event['ts'], 
                                    'duration': duration
                                })
                                self.last_typed_char_ts = press_event['ts']
                        else: # Not a char key (e.g. Ctrl, Enter), flush string buffer and log this click
                            self._flush_typed_char_buffer()
                            self._write_event(key_click_event)
                    else: # Press and release too far apart, log them separately
                        self._write_event(press_event)
                        self._write_event(event)
                else: # Release without a preceding press, log as is
                    self._write_event(event)
        
        # --- Mouse Event Processing ---
        elif device == "mouse":
            x, y = event['x'], event['y']
            if event_type == "click": # This is pynput's 'click' which is actually press/release
                button = event['button']
                pressed = event['pressed']
                pending_mouse_click_id = f"mouse_{button}_{x}_{y}" # crude ID, assumes no move during click

                if pressed: # Mouse button press
                    # If an unreleased press for this button, log it.
                    if pending_mouse_click_id in self.pending_events:
                         self._write_event(self.pending_events.pop(pending_mouse_click_id))
                    self.pending_events[pending_mouse_click_id] = event
                else: # Mouse button release
                    if pending_mouse_click_id in self.pending_events:
                        press_event = self.pending_events.pop(pending_mouse_click_id)
                        duration = round(ts - press_event['ts'], 5)
                        if duration <= MOUSE_CLICK_MAX_DELTA:
                             self._write_event({
                                "ts": press_event['ts'],
                                "device": "mouse",
                                "type": "mouse_click",
                                "x": x, "y": y, # Use release coordinates
                                "button": button,
                                "duration": duration
                            })
                        else: # Press and release too far apart
                            self._write_event(press_event)
                            self._write_event(event)
                    else: # Release without matching press
                        self._write_event(event)

            elif event_type == "move":
                if not self.mouse_move_buffer or \
                   (ts - self.last_mouse_move_ts <= MOUSE_SEQUENCE_MAX_DELTA):
                    self.mouse_move_buffer.append(event)
                else:
                    self._flush_mouse_move_buffer()
                    self.mouse_move_buffer.append(event) # Start new buffer
                self.last_mouse_move_ts = ts
            
            elif event_type == "scroll":
                # Optional: filter out (dx=0, dy=0) scrolls if not part of a sequence
                # if event['dx'] == 0 and event['dy'] == 0 and not self.mouse_scroll_buffer:
                #     # self._write_event(event) # or just skip
                #     return

                if not self.mouse_scroll_buffer or \
                   (ts - self.last_mouse_scroll_ts <= MOUSE_SEQUENCE_MAX_DELTA):
                    self.mouse_scroll_buffer.append(event)
                else:
                    self._flush_mouse_scroll_buffer()
                    self.mouse_scroll_buffer.append(event) # Start new buffer
                self.last_mouse_scroll_ts = ts
            
            else: # Unknown mouse event type
                self._write_event(event)
        
        else: # Unknown device
            self._write_event(event)

    def finalize(self):
        """Flush any remaining buffers and pending events."""
        self._flush_typed_char_buffer()
        self._flush_mouse_move_buffer()
        self._flush_mouse_scroll_buffer()
        
        # Log any remaining pending press events
        for pending_event_id in list(self.pending_events.keys()):
            self._write_event(self.pending_events.pop(pending_event_id))
            
        self.output_file.close()
        print(f"Compressed events written to {self.output_file.name}")


# --- Main Processing ---
if __name__ == "__main__":
    input_jsonl_path = "keystrokes.jsonl"  # Your input file
    output_jsonl_path = "keystrokes_compressed.jsonl" # Compressed output

    compressor = EventCompressor(output_jsonl_path)

    with open(input_jsonl_path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile):
            event = json.loads(line.strip())
            compressor.process_event(event)

    compressor.finalize()