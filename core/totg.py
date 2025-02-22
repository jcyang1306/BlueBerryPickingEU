import numpy as np
import math
from collections import deque

EPS = 0.000001
LOGNAME = "trajectory_processing.time_optimal_trajectory_generation"

class PathSegment:
    def __init__(self, length):
        self.length_ = length
        self.position_ = 0.0

    def getConfig(self, s):
        raise NotImplementedError

    def getTangent(self, s):
        raise NotImplementedError

    def getCurvature(self, s):
        raise NotImplementedError

    def getSwitchingPoints(self):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

class LinearPathSegment(PathSegment):
    def __init__(self, start, end):
        super().__init__(np.linalg.norm(end - start))
        self.end_ = end
        self.start_ = start

    def getConfig(self, s):
        s /= self.length_
        s = max(0.0, min(1.0, s))
        return (1.0 - s) * self.start_ + s * self.end_

    def getTangent(self, s):
        return (self.end_ - self.start_) / self.length_

    def getCurvature(self, s):
        return np.zeros_like(self.start_)

    def getSwitchingPoints(self):
        return []

    def clone(self):
        return LinearPathSegment(self.start_, self.end_)

class CircularPathSegment(PathSegment):
    def __init__(self, start, intersection, end, max_deviation):
        super().__init__(0.0)
        if np.linalg.norm(intersection - start) < EPS or np.linalg.norm(end - intersection) < EPS:
            self.length_ = 0.0
            self.radius = 1.0
            self.center = intersection
            self.x = np.zeros_like(start)
            self.y = np.zeros_like(start)
            return

        start_direction = (intersection - start) / np.linalg.norm(intersection - start)
        end_direction = (end - intersection) / np.linalg.norm(end - intersection)

        if np.linalg.norm(start_direction - end_direction) < EPS:
            self.length_ = 0.0
            self.radius = 1.0
            self.center = intersection
            self.x = np.zeros_like(start)
            self.y = np.zeros_like(start)
            return

        angle = math.acos(max(-1.0, start_direction.dot(end_direction)))
        start_distance = np.linalg.norm(start - intersection)
        end_distance = np.linalg.norm(end - intersection)

        distance = min(start_distance, end_distance)
        distance = min(distance, max_deviation * math.sin(0.5 * angle) / (1.0 - math.cos(0.5 * angle)))

        self.radius = distance / math.tan(0.5 * angle)
        self.length_ = angle * self.radius

        self.center = intersection + (end_direction - start_direction) / np.linalg.norm(end_direction - start_direction) * self.radius / math.cos(0.5 * angle)
        self.x = (intersection - distance * start_direction - self.center) / np.linalg.norm(intersection - distance * start_direction - self.center)
        self.y = start_direction

    def getConfig(self, s):
        angle = s / self.radius
        return self.center + self.radius * (self.x * math.cos(angle) + self.y * math.sin(angle))

    def getTangent(self, s):
        angle = s / self.radius
        return -self.x * math.sin(angle) + self.y * math.cos(angle)

    def getCurvature(self, s):
        angle = s / self.radius
        return -1.0 / self.radius * (self.x * math.cos(angle) + self.y * math.sin(angle))

    def getSwitchingPoints(self):
        switching_points = []
        dim = self.x.size
        for i in range(dim):
            switching_angle = math.atan2(self.y[i], self.x[i])
            if switching_angle < 0.0:
                switching_angle += math.pi
            switching_point = switching_angle * self.radius
            if switching_point < self.length_:
                switching_points.append(switching_point)
        switching_points.sort()
        return switching_points

    def clone(self):
        return CircularPathSegment(self.start_, self.center, self.end_, self.radius)

class Path:
    def __init__(self, path, max_deviation):
        self.length_ = 0.0
        self.path_segments_ = deque()
        self.switching_points_ = deque()

        if len(path) < 2:
            return

        path_iterator1 = iter(path)
        path_iterator2 = iter(path)
        next(path_iterator2)
        path_iterator3 = iter(path)
        next(path_iterator3, None)

        start_config = next(path_iterator1)
        for point2 in path_iterator2:
            point3 = next(path_iterator3, None)
            if max_deviation > 0.0 and point3 is not None:
                blend_segment = CircularPathSegment(0.5 * (start_config + point2), point2, 0.5 * (point2 + point3), max_deviation)
                end_config = blend_segment.getConfig(0.0)
                if np.linalg.norm(end_config - start_config) > EPS:
                    self.path_segments_.append(LinearPathSegment(start_config, end_config))
                self.path_segments_.append(blend_segment)
                start_config = blend_segment.getConfig(blend_segment.getLength())
            else:
                self.path_segments_.append(LinearPathSegment(start_config, point2))
                start_config = point2

        for segment in self.path_segments_:
            segment.position_ = self.length_
            local_switching_points = segment.getSwitchingPoints()
            for point in local_switching_points:
                self.switching_points_.append((self.length_ + point, False))
            self.length_ += segment.getLength()
            while self.switching_points_ and self.switching_points_[-1][0] >= self.length_:
                self.switching_points_.pop()
            self.switching_points_.append((self.length_, True))
        if self.switching_points_:
            self.switching_points_.pop()

    def getLength(self):
        return self.length_

    def getPathSegment(self, s):
        for segment in self.path_segments_:
            if s < segment.position_ + segment.length_:
                s -= segment.position_
                return segment
        return None

    def getConfig(self, s):
        path_segment = self.getPathSegment(s)
        return path_segment.getConfig(s)

    def getTangent(self, s):
        path_segment = self.getPathSegment(s)
        return path_segment.getTangent(s)

    def getCurvature(self, s):
        path_segment = self.getPathSegment(s)
        return path_segment.getCurvature(s)

    def getNextSwitchingPoint(self, s, discontinuity):
        for point, disc in self.switching_points_:
            if point > s:
                discontinuity[0] = disc
                return point
        discontinuity[0] = True
        return self.length_

    def getSwitchingPoints(self):
        return self.switching_points_

class TrajectoryStep:
    def __init__(self, path_pos, path_vel):
        self.path_pos_ = path_pos
        self.path_vel_ = path_vel
        self.time_ = 0.0

class Trajectory:
    def __init__(self, path, max_velocity, max_acceleration, time_step):
        self.path_ = path
        self.max_velocity_ = max_velocity
        self.max_acceleration_ = max_acceleration
        self.joint_num_ = len(max_velocity)
        self.valid_ = True
        self.time_step_ = time_step
        self.cached_time_ = float('inf')
        self.trajectory_ = deque([TrajectoryStep(0.0, 0.0)])
        self.end_trajectory_ = deque()

        after_acceleration = self.getMinMaxPathAcceleration(0.0, 0.0, True)
        while self.valid_ and not self.integrateForward(self.trajectory_, after_acceleration) and self.valid_:
            before_acceleration = 0.0
            switching_point = TrajectoryStep(0.0, 0.0)
            if self.getNextSwitchingPoint(self.trajectory_[-1].path_pos_, switching_point, before_acceleration, after_acceleration):
                break
            self.integrateBackward(self.trajectory_, switching_point.path_pos_, switching_point.path_vel_, before_acceleration)

        if self.valid_:
            before_acceleration = self.getMinMaxPathAcceleration(self.path_.getLength(), 0.0, False)
            self.integrateBackward(self.trajectory_, self.path_.getLength(), 0.0, before_acceleration)

        if self.valid_:
            previous = self.trajectory_[0]
            for step in self.trajectory_[1:]:
                step.time_ = previous.time_ + (step.path_pos_ - previous.path_pos_) / ((step.path_vel_ + previous.path_vel_) / 2.0)
                previous = step

    def isValid(self):
        return self.valid_

    def getDuration(self):
        return self.trajectory_[-1].time_

    def getTrajectorySegment(self, time):
        if time >= self.trajectory_[-1].time_:
            return self.trajectory_[-1]
        else:
            if time < self.cached_time_:
                self.cached_trajectory_segment_ = self.trajectory_[0]
            while time >= self.cached_trajectory_segment_.time_:
                self.cached_trajectory_segment_ += 1
            self.cached_time_ = time
            return self.cached_trajectory_segment_

    def getPosition(self, time):
        it = self.getTrajectorySegment(time)
        previous = it - 1

        time_step = it.time_ - previous.time_
        acceleration = 2.0 * (it.path_pos_ - previous.path_pos_ - time_step * previous.path_vel_) / (time_step * time_step)

        time_step = time - previous.time_
        path_pos = previous.path_pos_ + time_step * previous.path_vel_ + 0.5 * time_step * time_step * acceleration

        return self.path_.getConfig(path_pos)

    def getVelocity(self, time):
        it = self.getTrajectorySegment(time)
        previous = it - 1

        time_step = it.time_ - previous.time_
        acceleration = 2.0 * (it.path_pos_ - previous.path_pos_ - time_step * previous.path_vel_) / (time_step * time_step)

        time_step = time - previous.time_
        path_pos = previous.path_pos_ + time_step * previous.path_vel_ + 0.5 * time_step * time_step * acceleration
        path_vel = previous.path_vel_ + time_step * acceleration

        return self.path_.getTangent(path_pos) * path_vel

    def getAcceleration(self, time):
        it = self.getTrajectorySegment(time)
        previous = it - 1

        time_step = it.time_ - previous.time_
        acceleration = 2.0 * (it.path_pos_ - previous.path_pos_ - time_step * previous.path_vel_) / (time_step * time_step)

        time_step = time - previous.time_
        path_pos = previous.path_pos_ + time_step * previous.path_vel_ + 0.5 * time_step * time_step * acceleration
        path_vel = previous.path_vel_ + time_step * acceleration
        path_acc = (self.path_.getTangent(path_pos) * path_vel - self.path_.getTangent(previous.path_pos_) * previous.path_vel_)
        if time_step > 0.0:
            path_acc /= time_step
        return path_acc

    def getNextSwitchingPoint(self, path_pos, next_switching_point, before_acceleration, after_acceleration):
        acceleration_switching_point = TrajectoryStep(path_pos, 0.0)
        acceleration_before_acceleration = 0.0
        acceleration_after_acceleration = 0.0
        acceleration_reached_end = False
        while not acceleration_reached_end and acceleration_switching_point.path_vel_ > self.getVelocityMaxPathVelocity(acceleration_switching_point.path_pos_):
            acceleration_reached_end = self.getNextAccelerationSwitchingPoint(acceleration_switching_point.path_pos_, acceleration_switching_point, acceleration_before_acceleration, acceleration_after_acceleration)

        velocity_switching_point = TrajectoryStep(path_pos, 0.0)
        velocity_before_acceleration = 0.0
        velocity_after_acceleration = 0.0
        velocity_reached_end = False
        while not velocity_reached_end and (velocity_switching_point.path_pos_ <= acceleration_switching_point.path_pos_ and (velocity_switching_point.path_vel_ > self.getAccelerationMaxPathVelocity(velocity_switching_point.path_pos_ - EPS) or velocity_switching_point.path_vel_ > self.getAccelerationMaxPathVelocity(velocity_switching_point.path_pos_ + EPS))):
            velocity_reached_end = self.getNextVelocitySwitchingPoint(velocity_switching_point.path_pos_, velocity_switching_point, velocity_before_acceleration, velocity_after_acceleration)

        if acceleration_reached_end and velocity_reached_end:
            return True
        elif not acceleration_reached_end and (velocity_reached_end or acceleration_switching_point.path_pos_ <= velocity_switching_point.path_pos_):
            next_switching_point.path_pos_ = acceleration_switching_point.path_pos_
            next_switching_point.path_vel_ = acceleration_switching_point.path_vel_
            before_acceleration = acceleration_before_acceleration
            after_acceleration = acceleration_after_acceleration
            return False
        else:
            next_switching_point.path_pos_ = velocity_switching_point.path_pos_
            next_switching_point.path_vel_ = velocity_switching_point.path_vel_
            before_acceleration = velocity_before_acceleration
            after_acceleration = velocity_after_acceleration
            return False

    def getNextAccelerationSwitchingPoint(self, path_pos, next_switching_point, before_acceleration, after_acceleration):
        switching_path_pos = path_pos
        switching_path_vel = 0.0
        while True:
            discontinuity = [False]
            switching_path_pos = self.path_.getNextSwitchingPoint(switching_path_pos, discontinuity)

            if switching_path_pos > self.path_.getLength() - EPS:
                return True

            if discontinuity[0]:
                before_path_vel = self.getAccelerationMaxPathVelocity(switching_path_pos - EPS)
                after_path_vel = self.getAccelerationMaxPathVelocity(switching_path_pos + EPS)
                switching_path_vel = min(before_path_vel, after_path_vel)
                before_acceleration = self.getMinMaxPathAcceleration(switching_path_pos - EPS, switching_path_vel, False)
                after_acceleration = self.getMinMaxPathAcceleration(switching_path_pos + EPS, switching_path_vel, True)

                if (before_path_vel > after_path_vel or self.getMinMaxPhaseSlope(switching_path_pos - EPS, switching_path_vel, False) > self.getAccelerationMaxPathVelocityDeriv(switching_path_pos - 2.0 * EPS)) and (before_path_vel < after_path_vel or self.getMinMaxPhaseSlope(switching_path_pos + EPS, switching_path_vel, True) < self.getAccelerationMaxPathVelocityDeriv(switching_path_pos + 2.0 * EPS)):
                    break
            else:
                switching_path_vel = self.getAccelerationMaxPathVelocity(switching_path_pos)
                before_acceleration = 0.0
                after_acceleration = 0.0

                if self.getAccelerationMaxPathVelocityDeriv(switching_path_pos - EPS) < 0.0 and self.getAccelerationMaxPathVelocityDeriv(switching_path_pos + EPS) > 0.0:
                    break

        next_switching_point.path_pos_ = switching_path_pos
        next_switching_point.path_vel_ = switching_path_vel
        return False

    def getNextVelocitySwitchingPoint(self, path_pos, next_switching_point, before_acceleration, after_acceleration):
        step_size = 0.001
        accuracy = 0.000001

        start = False
        path_pos -= step_size
        while not (start and self.getMinMaxPhaseSlope(path_pos, self.getVelocityMaxPathVelocity(path_pos), False) <= self.getVelocityMaxPathVelocityDeriv(path_pos)) and path_pos < self.path_.getLength():
            path_pos += step_size
            if self.getMinMaxPhaseSlope(path_pos, self.getVelocityMaxPathVelocity(path_pos), False) >= self.getVelocityMaxPathVelocityDeriv(path_pos):
                start = True

        if path_pos >= self.path_.getLength():
            return True

        before_path_pos = path_pos - step_size
        after_path_pos = path_pos
        while after_path_pos - before_path_pos > accuracy:
            path_pos = (before_path_pos + after_path_pos) / 2.0
            if self.getMinMaxPhaseSlope(path_pos, self.getVelocityMaxPathVelocity(path_pos), False) > self.getVelocityMaxPathVelocityDeriv(path_pos):
                before_path_pos = path_pos
            else:
                after_path_pos = path_pos

        before_acceleration = self.getMinMaxPathAcceleration(before_path_pos, self.getVelocityMaxPathVelocity(before_path_pos), False)
        after_acceleration = self.getMinMaxPathAcceleration(after_path_pos, self.getVelocityMaxPathVelocity(after_path_pos), True)
        next_switching_point.path_pos_ = after_path_pos
        next_switching_point.path_vel_ = self.getVelocityMaxPathVelocity(after_path_pos)
        return False

    def integrateForward(self, trajectory, acceleration):
        path_pos = trajectory[-1].path_pos_
        path_vel = trajectory[-1].path_vel_

        switching_points = self.path_.getSwitchingPoints()
        next_discontinuity = iter(switching_points)
        next_discontinuity = next(next_discontinuity, None)

        while True:
            while next_discontinuity is not None and (next_discontinuity[0] <= path_pos or not next_discontinuity[1]):
                next_discontinuity = next(next_discontinuity, None)

            old_path_pos = path_pos
            old_path_vel = path_vel

            path_vel += self.time_step_ * acceleration
            path_pos += self.time_step_ * 0.5 * (old_path_vel + path_vel)

            if next_discontinuity is not None and path_pos > next_discontinuity[0]:
                if path_pos - next_discontinuity[0] < EPS:
                    continue
               




#     path_vel = old_path_vel + (next_discontinuity[0] - old_path_pos) * (path_vel - old_path_vel) / (path_pos - old_path_pos)
#     path_pos = next_discontinuity[0]
# else:
#     trajectory.append(TrajectoryStep(path_pos, path_vel))
#     if path_pos > self.path_.getLength():
#         return True
#     elif path_vel < 0.0:
#         self.valid_ = False
#         print(f"[{LOGNAME}] Error while integrating forward: Negative path velocity")
#         return True

#     if path_vel > self.getVelocityMaxPathVelocity(path_pos) and self.getMinMaxPhaseSlope(old_path_pos, self.getVelocityMaxPathVelocity(old_path_pos), False) <= self.getVelocityMaxPathVelocityDeriv(old_path_pos):
#         path_vel = self.getVelocityMaxPathVelocity(path_pos)

#     trajectory.append(TrajectoryStep(path_pos, path_vel))
#     acceleration = self.getMinMaxPathAcceleration(path_pos, path_vel, True)

#     if path_vel > self.getAccelerationMaxPathVelocity(path_pos) or path_vel > self.getVelocityMaxPathVelocity(path_pos):
#         # Find more accurate intersection with max-velocity curve using bisection
#         overshoot = trajectory.pop()
#         before = trajectory[-1].path_pos_
#         before_path_vel = trajectory[-1].path_vel_
#         after = overshoot.path_pos_
#         after_path_vel = overshoot.path_vel_
#         while after - before > EPS:
#             midpoint = 0.5 * (before + after)
#             midpoint_path_vel = 0.5 * (before_path_vel + after_path_vel)

#             if midpoint_path_vel > self.getVelocityMaxPathVelocity(midpoint) and self.getMinMaxPhaseSlope(before, self.getVelocityMaxPathVelocity(before), False) <= self.getVelocityMaxPathVelocityDeriv(before):
#                 midpoint_path_vel = self.getVelocityMaxPathVelocity(midpoint)

#             if midpoint_path_vel > self.getAccelerationMaxPathVelocity(midpoint) or midpoint_path_vel > self.getVelocityMaxPathVelocity(midpoint):
#                 after = midpoint
#                 after_path_vel = midpoint_path_vel
#             else:
#                 before = midpoint
#                 before_path_vel = midpoint_path_vel
#         trajectory.append(TrajectoryStep(before, before_path_vel))

#         if self.getAccelerationMaxPathVelocity(after) < self.getVelocityMaxPathVelocity(after):
#             if after > next_discontinuity[0]:
#                 return False
#             elif self.getMinMaxPhaseSlope(trajectory[-1].path_pos_, trajectory[-1].path_vel_, True) > self.getAccelerationMaxPathVelocityDeriv(trajectory[-1].path_pos_):
#                 return False
#         else:
#             if self.getMinMaxPhaseSlope(trajectory[-1].path_pos_, trajectory[-1].path_vel_, False) > self.getVelocityMaxPathVelocityDeriv(trajectory[-1].path_pos_):
#                 return False

def integrateBackward(self, start_trajectory, path_pos, path_vel, acceleration):
    start2 = iter(start_trajectory)
    start1 = iter(start_trajectory)
    next(start2, None)
    trajectory = deque()
    slope = 0.0
    assert start1 is not None and start1.path_pos_ <= path_pos

    while start1 is not None or path_pos >= 0.0:
        if start1 is not None and start1.path_pos_ <= path_pos:
            trajectory.appendleft(TrajectoryStep(path_pos, path_vel))
            path_vel -= self.time_step_ * acceleration
            path_pos -= self.time_step_ * 0.5 * (path_vel + trajectory[0].path_vel_)
            acceleration = self.getMinMaxPathAcceleration(path_pos, path_vel, False)
            slope = (trajectory[0].path_vel_ - path_vel) / (trajectory[0].path_pos_ - path_pos)

            if path_vel < 0.0:
                self.valid_ = False
                print(f"[{LOGNAME}] Error while integrating backward: Negative path velocity")
                self.end_trajectory_ = trajectory
                return
        else:
            start1 = next(start1, None)
            start2 = next(start2, None)

        # Check for intersection between current start trajectory and backward trajectory segments
        if start2 is not None:
            start_slope = (start2.path_vel_ - start1.path_vel_) / (start2.path_pos_ - start1.path_pos_)
            intersection_path_pos = (start1.path_vel_ - path_vel + slope * path_pos - start_slope * start1.path_pos_) / (slope - start_slope)
            if max(start1.path_pos_, path_pos) - EPS <= intersection_path_pos <= EPS + min(start2.path_pos_, trajectory[0].path_pos_):
                intersection_path_vel = start1.path_vel_ + start_slope * (intersection_path_pos - start1.path_pos_)
                start_trajectory = deque([step for step in start_trajectory if step.path_pos_ <= start1.path_pos_])
                start_trajectory.append(TrajectoryStep(intersection_path_pos, intersection_path_vel))
                start_trajectory.extend(trajectory)
                return

    self.valid_ = False
    print(f"[{LOGNAME}] Error while integrating backward: Did not hit start trajectory")
    self.end_trajectory_ = trajectory

def getMinMaxPathAcceleration(self, path_pos, path_vel, max):
    config_deriv = self.path_.getTangent(path_pos)
    config_deriv2 = self.path_.getCurvature(path_pos)
    factor = 1.0 if max else -1.0
    max_path_acceleration = float('inf')
    for i in range(self.joint_num_):
        if config_deriv[i] != 0.0:
            max_path_acceleration = min(max_path_acceleration, self.max_acceleration_[i] / abs(config_deriv[i]) - factor * config_deriv2[i] * path_vel * path_vel / config_deriv[i])
    return factor * max_path_acceleration

