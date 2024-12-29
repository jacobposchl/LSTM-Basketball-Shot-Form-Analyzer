# shot_detection.py

import math  # Import math module for mathematical operations
from config import (
    ShotState,  # Enum or constants representing different states of the shot detection state machine
    VELOCITY_THRESHOLD,  # Threshold for wrist velocity to detect significant movement
    CONSECUTIVE_FRAMES,  # Number of consecutive frames to consider for velocity history
    DISTANCE_THRESHOLD,  # Distance threshold to determine if the shot is valid
    TIME_THRESHOLD,  # Time threshold within which the shot should be completed
    VERTICAL_DISPLACEMENT_THRESHOLD,  # Threshold for vertical displacement to detect upward movement
    BALL_WRIST_DISTANCE_THRESHOLD,  # (Possibly unused) Threshold for distance between ball and wrist
    STABLE_FRAMES_REQUIRED,  # Number of stable frames required to transition state
    STABLE_VELOCITY_THRESHOLD,  # Velocity threshold to consider wrist as stable
    SHOT_COOLDOWN_FRAMES  # Number of frames to wait after detecting a shot before detecting another
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
        self.logger = logging.getLogger(__name__)  # Initialize a logger for the class
        self.shot_num = 0  # Counter for the number of shots detected
        self.shots = []  # List to store details of detected shots
        self.current_shot = None  # Dictionary to store details of the current shot being detected
        self.is_ball_close = None
        self.distance = None
        self.detect_hands = False
        self.make_shot = False

    def reset_shot_state(self):
        """
        Resets the shot detection state to its initial configuration.
        """
        self.state = ShotState.WAITING_FOR_STABILITY  # Set the initial state to WAITING_FOR_STABILITY
        self.stable_frames = 0  # Counter for the number of consecutive stable frames
        self.velocity_history = deque(maxlen=CONSECUTIVE_FRAMES)  # Queue to store recent wrist velocities
        self.baseline_wrist_y = None  # Baseline Y-coordinate of the wrist for displacement calculation
        self.cooldown_counter = 0  # Counter for cooldown frames after a shot is detected

    def update(self, wrist_vel, wrist_abs_y, wrist_pos, ball_pos, fps, frame_count):
        """
        Updates the shot detection state based on current wrist velocity and ball proximity.

        Args:
            wrist_vel (float): Current velocity of the wrist.
            is_ball_close (bool): Whether the ball is close to the wrist.
            wrist_abs_y (float): Y-coordinate of the wrist.
            wrist_pos (tuple): (x, y) coordinates of the wrist in the original frame.
            ball_pos (tuple): (x, y) coordinates of the ball in the original frame.
            fps (float): Frames per second of the video.
            frame_count (int): Current frame number.

        Returns:
            None
        """
        # Log the current state and relevant parameters for debugging
        self.logger.debug(
            f"Frame {frame_count}: State={self.state}, Stable Frames={self.stable_frames}, "
            f"Wrist Vel={wrist_vel}, Wrist Abs Y={wrist_abs_y}"
        )

        # Calculate the Euclidean distance between the wrist and the ball
        self.distance = self.calculate_distance(wrist_pos, ball_pos)
        if self.distance is not None:
            self.is_ball_close = self.distance <= BALL_WRIST_DISTANCE_THRESHOLD
        else:
            self.is_ball_close = False

        #detect_hands = True if ShotState = Cooldown or WaitingForStability, else False
        if self.state == ShotState.COOLDOWN or self.state == ShotState.WAITING_FOR_STABILITY:
            self.detect_hands = True
        else:
            self.detect_hands = False

        # Handle different states of the state machine
        if self.state == ShotState.WAITING_FOR_STABILITY:
            # Check if wrist velocity is below the stable threshold and the ball is close
            if wrist_vel is not None and wrist_vel < STABLE_VELOCITY_THRESHOLD and self.is_ball_close:
                self.stable_frames += 1  # Increment stable frame counter
                self.logger.debug(
                    f"Stable frames increased to {self.stable_frames} "
                    f"(wrist_vel={wrist_vel}, is_ball_close={self.is_ball_close})"
                )
            else:
                # If stability or ball proximity is lost, reset the stable frame counter
                if self.stable_frames != 0:
                    self.logger.debug("Stability or ball proximity lost. Resetting stable_frames.")
                self.stable_frames = 0

            # If enough stable frames are detected, transition to READY_TO_DETECT_SHOT state
            if self.stable_frames >= STABLE_FRAMES_REQUIRED:
                self.baseline_wrist_y = wrist_abs_y  # Set baseline Y-coordinate for displacement
                self.state = ShotState.READY_TO_DETECT_SHOT  # Transition state
                self.velocity_history.clear()  # Clear velocity history for new detection
                self.logger.info(f"Baseline wrist Y set at {self.baseline_wrist_y}")
                self.logger.info("Transitioned to READY_TO_DETECT_SHOT")

        elif self.state == ShotState.READY_TO_DETECT_SHOT:
            # Append current wrist velocity to the history, using 0 if velocity is None
            if wrist_vel is not None:
                self.velocity_history.append(wrist_vel)
                self.logger.debug(f"Appended wrist_vel={wrist_vel} to velocity_history")
            else:
                self.velocity_history.append(0)
                self.logger.debug("Appended wrist_vel=0 to velocity_history")

            # Check if enough velocity data has been collected
            if len(self.velocity_history) == CONSECUTIVE_FRAMES:
                # Check if all recent velocities exceed the defined threshold
                if all(v > VELOCITY_THRESHOLD for v in self.velocity_history):
                    # Calculate vertical displacement from baseline
                    displacement = self.baseline_wrist_y - wrist_abs_y
                    self.logger.debug(f"Displacement={displacement}")

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
                        self.velocity_history.clear()  # Clear velocity history for next detection
                        self.logger.info(f"Shot {self.shot_num} detected. Transitioned to SHOT_IN_PROGRESS.")
                    else:
                        # If displacement is insufficient, do not count as shot
                        self.logger.debug(
                            "High velocity detected but no upward displacement. Not counting as shot start."
                        )
                        self.velocity_history.clear()

        elif self.state == ShotState.SHOT_IN_PROGRESS:
            if not self.is_ball_close:
                # Calculate the duration of the shot so far
                shot_duration = (frame_count / fps) - self.current_shot['start_time']
                if self.distance:

                    # Check if the ball has moved away beyond the distance threshold within the time threshold
                    if self.distance > DISTANCE_THRESHOLD and shot_duration <= TIME_THRESHOLD:
                        # Valid shot end detected
                        self.current_shot['end_frame'] = frame_count  # Record end frame
                        self.current_shot['end_time'] = frame_count / fps  # Record end time
                        self.current_shot['duration'] = shot_duration  # Record shot duration
                        self.shots.append(self.current_shot)  # Add shot to the list of detected shots
                        self.logger.info(
                            f"Shot {self.current_shot['shot_id']} ended at frame {frame_count} "
                            f"(time {self.current_shot['end_time']:.2f}s) with duration {shot_duration:.2f}s"
                        )
                        self.current_shot = None  # Reset current shot
                        self.state = ShotState.COOLDOWN  # Transition to COOLDOWN state
                        self.cooldown_counter = SHOT_COOLDOWN_FRAMES  # Initialize cooldown counter
                    elif shot_duration > TIME_THRESHOLD:
                        # If shot duration exceeds time threshold without valid end, invalidate the shot
                        self.current_shot['invalid'] = True  # Mark shot as invalid
                        self.shots.append(self.current_shot)  # Add invalid shot to the list
                        self.logger.info(
                            f"Shot {self.current_shot['shot_id']} invalidated due to duration "
                            f"{shot_duration:.2f}s exceeding TIME_THRESHOLD."
                        )
                        self.current_shot = None  # Reset current shot
                        self.state = ShotState.COOLDOWN  # Transition to COOLDOWN state
                        self.cooldown_counter = SHOT_COOLDOWN_FRAMES  # Initialize cooldown counter
                else:
                    # Distance calculation failed
                    pass
            else:
                # If the ball is still close, check if the time threshold has been exceeded
                shot_duration = (frame_count / fps) - self.current_shot['start_time']  # Calculate shot duration
                if shot_duration > TIME_THRESHOLD:
                    # Invalidate the shot if it exceeds the time threshold without completing
                    self.current_shot['invalid'] = True  # Mark shot as invalid
                    self.shots.append(self.current_shot)  # Add invalid shot to the list
                    self.logger.info(
                        f"Shot {self.current_shot['shot_id']} invalidated due to exceeding "
                        f"TIME_THRESHOLD without detecting end."
                    )
                    self.current_shot = None  # Reset current shot
                    self.state = ShotState.COOLDOWN  # Transition to COOLDOWN state
                    self.cooldown_counter = SHOT_COOLDOWN_FRAMES  # Initialize cooldown counter

        elif self.state == ShotState.COOLDOWN:
            # Decrement the cooldown counter each frame
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                # Once cooldown is complete, reset the state machine to WAITING_FOR_STABILITY
                self.reset_shot_state()
                self.logger.info("Cooldown complete. Transitioned to WAITING_FOR_STABILITY.")

    def assign_make_shot(self, make_shot):
        """
        Assigns the make_shot variable to the class attribute.

        Args:
            make_shot (bool): Whether the shot was made.

        Returns:
            None
        """

        # Iterate over the shots list in reverse to find the last valid shot
        for shot in reversed(self.shots):
            if not shot.get('invalid', True):
                shot['make'] = make_shot
                return  # Exit after assigning to the last valid shot


    def calculate_distance(self, wrist_pos, ball_pos):
        """
        Calculates the Euclidean distance between the wrist and the ball.

        Args:
            wrist_pos (tuple): (x, y) coordinates of the wrist in the original frame.
            ball_pos (tuple): (x, y) coordinates of the ball in the original frame.

        Returns:
            float: Euclidean distance between wrist and ball.
        """
        # Check if either wrist position or ball position is None
        if wrist_pos is None or ball_pos is None:
            self.logger.warning("Wrist position or ball position is None. Returning infinite distance.")
            return None  # Return None to indicate invalid distance

        # Calculate Euclidean distance using the formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
        distance = math.sqrt(
            (wrist_pos[0] - ball_pos[0]) ** 2 + 
            (wrist_pos[1] - ball_pos[1]) ** 2
        )
        return distance  # Return the calculated distance

    def get_shots(self):
        """
        Returns the list of detected shots.

        Returns:
            list: List of shot dictionaries containing shot details.
        """
        return self.shots  # Return the list of detected shots
