# shot_detection.py

import math  # Import math module for mathematical operations
from config import (
    ShotState,  # Enum or constants representing different states of the shot detection state machine
    VELOCITY_THRESHOLD,  # Threshold for wrist velocity to detect significant movement
    CONSECUTIVE_FRAMES,  # Number of consecutive frames to consider for velocity history
    DISTANCE_THRESHOLD,  # Distance threshold to determine if the shot is valid
    TIME_THRESHOLD,  # Time threshold within which the shot should be completed
    VERTICAL_DISPLACEMENT_THRESHOLD,  # Threshold for vertical displacement to detect upward movement
    BALL_WRIST_DISTANCE_THRESHOLD,  # Threshold for distance between ball and wrist
    STABLE_FRAMES_REQUIRED,  # Number of stable frames required to transition state
    STABLE_VELOCITY_THRESHOLD,  # Velocity threshold to consider wrist as stable
    SHOT_COOLDOWN_FRAMES,  # Number of frames to wait after detecting a shot before detecting another
    DOMINANT_HAND,  # Dominant hand configuration
    PROBABILITY_INCREMENT_BALL_CLOSE,
    MIN_PROBABILITY,
    MAX_PROBABILITY,
    PROBABILITY_INCREMENT_WRISTS_CLOSE,
    PROBABILITY_DECREMENT_UNSTABLE,
    WRIST_CLOSE_DISTANCE_THRESHOLD,
    MAX_BALL_INVISIBLE_FRAMES,
    STABILITY_THRESHOLD,
    MAX_UNSTABLE_FRAMES,
    MAX_WAITING_UNSTABLE_FRAMES,
    ARM_STABLE_FRAMES,
    REBASELINE_STABLE_FRAMES
    )
from collections import deque  # Import deque for efficient queue operations
import logging  # Import logging module for logging events and debugging

class ShotDetector:
    """ 
    A class to handle shot detection using a state machine.
    """
    def __init__(self):
        """
        Initializes the ShotDetector with default state and configurations.
        """
        self.reset_shot_state()  # Reset the state machine to its initial state
        self.stability_frame = None 
        self.armed_stable_frames = 0
        self.logger = logging.getLogger(__name__)  # Initialize a logger for the class
        self.shot_num = 0  # Counter for the number of shots detected
        self.shots = []  # List to store details of detected shots
        self.current_shot = None  # Dictionary to store details of the current shot being detected
        self.is_ball_close = None
        self.distance = None
        self.detect_hands = False
        self.make_shot = False
        self.ball_stability_prob = 0
        self.post_ball_stability_prob       = 0.0
        self.post_stable_frames             = 0
        self.post_waiting_unstable_frames   = 0

        # Add these new attributes to track ball visibility and release
        self.ball_visible = False
        self.ball_was_released = False
        self.ball_released_and_disappeared = False
        self.last_ball_visible_frame = 0
        self.frames_without_ball = 0
        self.ball_disappeared_during_shot = False
        self.last_ball_velocity = None
        self.consecutive_invisible_frames = 0
        self.MIN_SHOT_FRAMES = 5  # Minimum frames for a shot to be considered valid
        self.MIN_RELEASE_DISTANCE = BALL_WRIST_DISTANCE_THRESHOLD * 1.5  # Distance at which we consider ball released
        self.MIN_BALL_VELOCITY_FOR_RELEASE = 180.0  # Minimum velocity to consider a ball as released

        # Initialize separate velocity histories for both wrists
        self.velocity_history_left = deque(maxlen=CONSECUTIVE_FRAMES)  # Left wrist velocity history
        self.velocity_history_right = deque(maxlen=CONSECUTIVE_FRAMES)  # Right wrist velocity history

    def reset_shot_state(self):
        """
        Resets the shot detection state to its initial configuration.
        """

        self.state = ShotState.WAITING_FOR_STABILITY  # Set the initial state to WAITING_FOR_STABILITY
        self.stable_frames = 0  # Counter for the number of consecutive stable frames
        self.stability_frame = None  # Counter for the number of stable frames
        # Removed self.velocity_history; now using separate histories for left and right
        self.baseline_wrist_y = None  # Baseline Y-coordinate of the wrist for displacement calculation
        self.cooldown_counter = 0  # Counter for cooldown frames after a shot is detected

        self.wrist_distance = None       
        self.are_wrists_close = False 

        self.waiting_unstable_frames = 0

        self.unstable_frames = 0

        # Reset ball tracking flags
        self.ball_visible = False
        self.ball_was_released = False
        self.ball_released_and_disappeared = False
        self.ball_disappeared_during_shot = False
        self.consecutive_invisible_frames = 0
        self.last_ball_velocity = None

    def update(self, left_wrist_vel, right_wrist_vel, left_wrist_abs_pos, right_wrist_abs_pos,
               ball_pos, fps, frame_count, ball_velocity=None):
        
        # Update ball velocity
        self.last_ball_velocity = ball_velocity
        
        # Update ball visibility status
        previous_ball_visible = self.ball_visible
        self.ball_visible = ball_pos is not None
        
        if self.state == ShotState.WAITING_FOR_STABILITY:
        # 1) track instability → reset only if sustained
            if self.ball_stability_prob < STABILITY_THRESHOLD:
                self.unstable_frames += 1
            else:
                self.unstable_frames = 0

            if self.unstable_frames >= MAX_UNSTABLE_FRAMES:
                # too much wobble before a shot ever came → start over
                self.reset_shot_state()
            else:
                # 2) update the probability of being stable
                self.update_ball_stability_prob(
                    left_wrist_vel, right_wrist_vel,
                    self.is_ball_close, ball_pos,
                    left_wrist_abs_pos, right_wrist_abs_pos,
                    self.wrist_distance, self.are_wrists_close
                )

        # Track consecutive frames with ball invisible
        if not self.ball_visible:
            self.consecutive_invisible_frames += 1
            self.frames_without_ball += 1
        else:
            self.consecutive_invisible_frames = 0
            self.last_ball_visible_frame = frame_count
            self.frames_without_ball = 0
        
        # Determine dominant wrist position based on configuration
        if DOMINANT_HAND == "RIGHT":
            dom_wrist_pos = right_wrist_abs_pos
        else:
            dom_wrist_pos = left_wrist_abs_pos 

        # Calculate distance between wrists if both are present
        if left_wrist_abs_pos is not None and right_wrist_abs_pos is not None:
            self.wrist_distance = self.calculate_distance(left_wrist_abs_pos, right_wrist_abs_pos)
        else:
            self.wrist_distance = None

        # Calculate distance between dominant wrist and ball
        self.distance = self.calculate_distance(dom_wrist_pos, ball_pos)
        
        # Determine if ball is close to wrist
        if self.distance is not None:
            self.is_ball_close = self.distance <= BALL_WRIST_DISTANCE_THRESHOLD
        else:
            self.is_ball_close = False

        # Determine if wrists are close based on threshold
        if self.wrist_distance is not None:
            self.are_wrists_close = self.wrist_distance <= WRIST_CLOSE_DISTANCE_THRESHOLD
        else:
            self.are_wrists_close = False

        
        # Detect hands only in specific states
        if self.state in [ShotState.COOLDOWN, ShotState.WAITING_FOR_STABILITY]:
            self.detect_hands = True
        else:
            self.detect_hands = False

        # Track if ball disappears during shot with release detection
        if self.state == ShotState.SHOT_IN_PROGRESS and previous_ball_visible and not self.ball_visible:
            # Check if the ball was likely released (away from wrist) when it disappeared
            if self.distance is not None and self.distance > self.MIN_RELEASE_DISTANCE:
                # Ball was already moving away from wrist when it disappeared
                self.ball_released_and_disappeared = True
                self.logger.debug(f"Ball released and disappeared at frame {frame_count}, distance: {self.distance:.2f}")
            elif ball_velocity is not None and ball_velocity > self.MIN_BALL_VELOCITY_FOR_RELEASE:
                # Ball had significant velocity when it disappeared
                self.ball_released_and_disappeared = True
                self.logger.debug(f"Ball with high velocity ({ball_velocity:.2f}) disappeared at frame {frame_count}")
            else:
                # Just a momentary detection issue while ball is still controlled
                self.ball_disappeared_during_shot = True
                self.logger.debug(f"Ball momentarily disappeared during shot at frame {frame_count}")

        # Handle different states of the state machine
        if self.state == ShotState.WAITING_FOR_STABILITY:
            self.logger.debug(f"ball stable prob: {self.ball_stability_prob} at frame {frame_count}")
            if self.ball_stability_prob >= STABILITY_THRESHOLD:
                self.stable_frames += 1
                self.waiting_unstable_frames = 0
            else:
                self.waiting_unstable_frames += 1
                if (self.waiting_unstable_frames >= MAX_WAITING_UNSTABLE_FRAMES):
                    self.stable_frames = 0
                    self.waiiting_unstable_frames = 0
            
            if self.stable_frames >= STABLE_FRAMES_REQUIRED:
                self.baseline_wrist_y = dom_wrist_pos[1]
                self.stability_frame = frame_count
                self.state = ShotState.READY_TO_DETECT_SHOT
                self.velocity_history_left.clear()
                self.velocity_history_right.clear()
                self.ball_stability_prob = 0  # Reset after transition
                self.logger.debug(f"Transition to READY_TO_DETECT_SHOT at frame {frame_count}")

        elif self.state == ShotState.READY_TO_DETECT_SHOT:

            # ——— post-stability check ———
            self.update_post_ball_stability_prob(left_wrist_vel, right_wrist_vel)

            if self.post_ball_stability_prob >= STABILITY_THRESHOLD:
                self.post_stable_frames += 1
                self.post_waiting_unstable_frames = 0
            else:
                self.post_waiting_unstable_frames += 1
                if self.post_waiting_unstable_frames >= MAX_WAITING_UNSTABLE_FRAMES:
                    self.post_stable_frames = 0
                    self.post_waiting_unstable_frames = 0

            # If we’ve seen enough post-stability, recalibrate the baseline
            if self.post_stable_frames >= STABLE_FRAMES_REQUIRED:
                self.logger.info(f"Re-baselining wrist at frame {frame_count}")
                self.baseline_wrist_y = dom_wrist_pos[1]   # new “start” height
                # reset the post-stability counters
                self.post_ball_stability_prob     = 0.0
                self.post_stable_frames           = 0
                self.post_waiting_unstable_frames = 0
            # ————————————————


            # Append current wrist velocities to the histories, using 0 if velocity is None
            if left_wrist_vel is not None:
                self.velocity_history_left.append(left_wrist_vel)
            else:
                self.velocity_history_left.append(0)

            if right_wrist_vel is not None:
                self.velocity_history_right.append(right_wrist_vel)
            else:
                self.velocity_history_right.append(0)

            # Check if enough velocity data has been collected
            if len(self.velocity_history_left) == CONSECUTIVE_FRAMES and len(self.velocity_history_right) == CONSECUTIVE_FRAMES:
                # Check if all recent left wrist velocities exceed the defined threshold
                left_velocity_exceeds = all(v > VELOCITY_THRESHOLD for v in self.velocity_history_left)
                # Check if all recent right wrist velocities exceed the defined threshold
                right_velocity_exceeds = all(v > VELOCITY_THRESHOLD for v in self.velocity_history_right)

                if left_velocity_exceeds and right_velocity_exceeds:
                    # Calculate vertical displacement from baseline
                    displacement = self.baseline_wrist_y - dom_wrist_pos[1]

                    # If displacement exceeds threshold, a shot is detected
                    if displacement > VERTICAL_DISPLACEMENT_THRESHOLD:
                        self.shot_num += 1  # Increment shot counter
                        # Initialize current shot details
                        self.current_shot = {
                            'start_frame': frame_count,
                            'end_frame': None,
                            'start_time': frame_count / fps,
                            'shot_id': self.shot_num,
                            'invalid': False,
                            'make': self.make_shot
                        }
                        self.state = ShotState.SHOT_IN_PROGRESS  # Transition to SHOT_IN_PROGRESS state
                        self.velocity_history_left.clear()
                        self.velocity_history_right.clear()  # Clear velocity histories for next detection
                        
                        # Reset ball tracking flags at the start of a new shot
                        self.ball_disappeared_during_shot = False
                        self.ball_was_released = False
                        self.ball_released_and_disappeared = False
                        
                        self.logger.debug(f"Shot {self.shot_num} detected at frame {frame_count}")
                    else:
                        self.velocity_history_left.clear()
                        self.velocity_history_right.clear()

        elif self.state == ShotState.SHOT_IN_PROGRESS:
            # Calculate the duration of the shot so far
            shot_duration = (frame_count / fps) - self.current_shot['start_time']
            shot_frames = frame_count - self.current_shot['start_frame']
            
            # Check for normal shot completion (ball moved away)
            if not self.is_ball_close and self.ball_visible:
                if self.distance and self.distance > DISTANCE_THRESHOLD and shot_duration <= TIME_THRESHOLD:
                    # Mark ball as released (useful if it disappears later)
                    self.ball_was_released = True
                    
                    # Valid shot end detected - normal case where ball is visible and distant
                    self.current_shot['end_frame'] = frame_count
                    self.current_shot['end_time'] = frame_count / fps
                    self.current_shot['duration'] = shot_duration
                    self.current_shot['detection_method'] = "normal"
                    self.shots.append(self.current_shot)
                    self.current_shot = None
                    self.state = ShotState.COOLDOWN
                    self.cooldown_counter = SHOT_COOLDOWN_FRAMES
                    self.logger.debug(f"Shot {self.shot_num} completed at frame {frame_count} (normal detection)")
                    
            # Alternative: Ball was released and then disappeared from frame
            elif self.ball_released_and_disappeared and shot_frames >= self.MIN_SHOT_FRAMES and shot_duration <= TIME_THRESHOLD:
                # Check if enough frames have passed since ball disappeared to ensure it's not just a momentary detection issue
                frames_since_disappearance = frame_count - self.last_ball_visible_frame
                if frames_since_disappearance >= 3:  # Allow a few frames to ensure it's not just a momentary detection issue
                    # Consider shot completed if ball disappeared after release
                    self.current_shot['end_frame'] = frame_count
                    self.current_shot['end_time'] = frame_count / fps
                    self.current_shot['duration'] = shot_duration
                    self.current_shot['detection_method'] = "ball_disappeared_after_release"
                    self.shots.append(self.current_shot)
                    self.current_shot = None
                    self.state = ShotState.COOLDOWN
                    self.cooldown_counter = SHOT_COOLDOWN_FRAMES
                    self.logger.debug(f"Shot {self.shot_num} completed at frame {frame_count} (ball released and disappeared)")
            
            # Alternative: Ball disappeared, but was moving away from wrist with high velocity
            elif self.consecutive_invisible_frames >= 5 and shot_frames >= self.MIN_SHOT_FRAMES and shot_duration <= TIME_THRESHOLD:
                # If ball has been invisible for several frames and the last known velocity was high
                if self.last_ball_velocity is not None and self.last_ball_velocity > self.MIN_BALL_VELOCITY_FOR_RELEASE:
                    self.current_shot['end_frame'] = frame_count
                    self.current_shot['end_time'] = frame_count / fps
                    self.current_shot['duration'] = shot_duration
                    self.current_shot['detection_method'] = "high_velocity_then_disappeared"
                    self.shots.append(self.current_shot)
                    self.current_shot = None
                    self.state = ShotState.COOLDOWN
                    self.cooldown_counter = SHOT_COOLDOWN_FRAMES
                    self.logger.debug(f"Shot {self.shot_num} completed at frame {frame_count} (high velocity then disappeared)")
            
            # Check for timeout
            elif shot_duration > TIME_THRESHOLD:
                # Invalidate the shot if it exceeds the time threshold without completing
                self.current_shot['invalid'] = True
                self.current_shot['detection_method'] = "timeout"
                self.shots.append(self.current_shot)
                self.current_shot = None
                self.state = ShotState.COOLDOWN
                self.cooldown_counter = SHOT_COOLDOWN_FRAMES
                self.logger.debug(f"Shot {self.shot_num} invalidated due to exceeding time threshold at frame {frame_count}")

        elif self.state == ShotState.COOLDOWN:
            # Decrement the cooldown counter each frame
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                # Once cooldown is complete, reset the state machine to WAITING_FOR_STABILITY
                self.reset_shot_state()
                self.logger.debug(f"Cooldown complete. Resetting shot detector at frame {frame_count}")

    def assign_make_shot(self, make_shot):
        """
        Assigns the make status to the most recent valid shot.

        Args:
            make_shot (bool): True if the shot was made, False otherwise.
        """
        # Iterate over the shots list in reverse to find the last valid shot
        for shot in reversed(self.shots):
            if not shot.get('invalid', True):
                shot['make'] = make_shot
                self.logger.debug(f"Assigned make_shot={make_shot} to shot_id={shot.get('shot_id')}")
                return  # Exit after assigning to the last valid shot

    def calculate_distance(self, wrist_pos, ball_pos):
        """
        Calculates the Euclidean distance between the wrist and the ball.

        Args:
            wrist_pos (tuple): (x, y) position of the wrist.
            ball_pos (tuple): (x, y) position of the ball.

        Returns:
            float: Euclidean distance between the wrist and the ball.
        """
        if wrist_pos is None or ball_pos is None:
            return None  # Return None to indicate invalid distance

        # Calculate Euclidean distance using the formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
        distance = math.sqrt(
            (wrist_pos[0] - ball_pos[0]) ** 2 + 
            (wrist_pos[1] - ball_pos[1]) ** 2
        )
        return distance  # Return the calculated distance

    def update_ball_stability_prob(self,
            left_wrist_vel, right_wrist_vel,
            is_ball_close, ball_pos,
            left_wrist_abs_pos, right_wrist_abs_pos,
            wrist_distance, are_wrists_close):
        """
        Simplified stability logic:
          • If ball is detected close AND both wrist velocities are below threshold → increment “ball-close” probability
          • Elif ball has been invisible for ≤ MAX_BALL_INVISIBLE_FRAMES AND both wrist velocities < threshold → increment “wrist-only” probability
          • Else → reset probability to zero
        """
        # Determine if wrists are “still”
        left_ok  = (left_wrist_vel  is not None and left_wrist_vel  < STABLE_VELOCITY_THRESHOLD)
        right_ok = (right_wrist_vel is not None and right_wrist_vel < STABLE_VELOCITY_THRESHOLD)

        # Allow a few frames of missing ball detections
        ball_invisible_ok = (self.consecutive_invisible_frames <= MAX_BALL_INVISIBLE_FRAMES)

        if is_ball_close and left_ok and right_ok:
            # Ball in hand & wrists still → strong increment
            self.ball_stability_prob = min(
                self.ball_stability_prob + PROBABILITY_INCREMENT_BALL_CLOSE,
                MAX_PROBABILITY
            )

        elif ball_invisible_ok and left_ok and right_ok:
            # Ball may be missed but wrists still → weaker increment
            self.ball_stability_prob = min(
                self.ball_stability_prob + PROBABILITY_INCREMENT_WRISTS_CLOSE,
                MAX_PROBABILITY
            )

        else:
            # Any other case → drop all stability
            self.ball_stability_prob = 0.0

    def update_post_ball_stability_prob(self, left_vel, right_vel):
        left_ok  = left_vel  is not None and left_vel  < STABLE_VELOCITY_THRESHOLD
        right_ok = right_vel is not None and right_vel < STABLE_VELOCITY_THRESHOLD

        # Always treat “ball invisible” as wrists-only
        if left_ok and right_ok:
            self.post_ball_stability_prob = min(
                self.post_ball_stability_prob + (PROBABILITY_INCREMENT_WRISTS_CLOSE * 1.5),
                MAX_PROBABILITY
            )
        else:
            self.post_ball_stability_prob = 0.0


    def get_shots(self):
        """
        Returns the list of detected shots.

        Returns:
            list: List of shot dictionaries containing shot details.
        """
        return self.shots  # Return the list of detected shots