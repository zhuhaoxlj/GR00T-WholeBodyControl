/**
 * @file input_interface.hpp
 * @brief Abstract base class for all input sources that can drive the G1 robot.
 *
 * InputInterface defines the polymorphic contract that every concrete input
 * handler (keyboard, gamepad, ZMQ, ROS2, …) must implement.  Two pure-virtual
 * methods form the core of the interface:
 *
 *   1. update()       – Poll the hardware / network for new data and latch it
 *                        into internal flags (called once per control-loop tick).
 *   2. handle_input() – Translate the latched flags into system-state changes
 *                        (motion switching, planner commands, heading adjustments,
 *                         operator start / stop, etc.).
 *
 * The base class also provides:
 *   - Thread-safe DataBuffer members for VR 3-point tracking, upper-body
 *     joint targets, hand joint targets, and external token state.
 *   - Shared stdin buffering so the InterfaceManager can pre-read keys and
 *     dispatch them to the correct active interface.
 *   - A safety-reset mechanism used by managers when switching active
 *     interfaces to prevent stale commands from carrying over.
 *   - Keyboard-controlled compliance and hand-close-ratio helpers that are
 *     shared across all input modes (g/h/b/v and x/c keys).
 *
 * @note The class is non-copyable and non-movable by design.  Instances are
 *       always managed via std::unique_ptr by the owning manager.
 */

#ifndef INPUT_INTERFACE_HPP
#define INPUT_INTERFACE_HPP

#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <iostream>
#include <queue>
#include <memory>
#include <optional>
#include <string>
#include "../utils.hpp"              // For DataBuffer  
#include "../robot_parameters.hpp"   // For HeadingState, OperatorState
#include "../motion_data_reader.hpp" // For MotionDataReader, MotionSequence
#include "../math_utils.hpp"         // For float_to_double
#include "../localmotion_kplanner.hpp" // For PlannerState, MovementState

namespace input_interface_motion_transition_detail {

constexpr int kGentleMotionTransitionFrames = 100;  // 2.0 s at the 50 Hz control rate.

inline double CalculateSmoothStepRatio(double raw_ratio) {
    double clamped_ratio = std::clamp(raw_ratio, 0.0, 1.0);
    return clamped_ratio * clamped_ratio * (3.0 - 2.0 * clamped_ratio);
}

inline double InterpolateScalar(double start_value, double end_value, double blend_ratio) {
    return start_value + (end_value - start_value) * blend_ratio;
}

}  // namespace input_interface_motion_transition_detail

inline std::shared_ptr<const MotionSequence> CreateGentleReferenceMotionTransition(
    const std::shared_ptr<const MotionSequence>& source_motion,
    int source_frame,
    const std::shared_ptr<const MotionSequence>& target_motion,
    bool append_target_motion = true,
    const std::string& transition_motion_name_prefix = "transition_motion:") {
    using namespace input_interface_motion_transition_detail;

    if (!target_motion || target_motion->timesteps <= 0) {
        return target_motion;
    }

    if (!source_motion || source_motion->timesteps <= 0) {
        return target_motion;
    }

    const int normalized_source_frame = std::clamp(source_frame, 0, source_motion->timesteps - 1);
    const int transition_frame_count = kGentleMotionTransitionFrames;
    const int appended_target_frame_count = append_target_motion ? target_motion->timesteps : 0;
    const int total_frame_count = transition_frame_count + appended_target_frame_count;

    auto transition_motion = std::make_shared<MotionSequence>();
    transition_motion->name = transition_motion_name_prefix + target_motion->name;
    transition_motion->timesteps = total_frame_count;
    transition_motion->encode_mode = target_motion->GetEncodeMode();
    transition_motion->SetBodyPartIndexes(target_motion->BodyPartIndexes());
    transition_motion->ReserveCapacity(total_frame_count,
                                       target_motion->GetNumJoints(),
                                       target_motion->GetNumBodies(),
                                       target_motion->GetNumBodyQuaternions(),
                                       target_motion->GetNumSmplJoints(),
                                       target_motion->GetNumSmplPoses());

    const int joint_count = target_motion->GetNumJoints();
    const int body_count = target_motion->GetNumBodies();
    const int body_quaternion_count = target_motion->GetNumBodyQuaternions();
    const int smpl_joint_count = target_motion->GetNumSmplJoints();
    const int smpl_pose_count = target_motion->GetNumSmplPoses();
    const int common_joint_count = std::min(source_motion->GetNumJoints(), joint_count);
    const int common_body_count = std::min(source_motion->GetNumBodies(), body_count);
    const int common_body_quaternion_count = std::min(source_motion->GetNumBodyQuaternions(), body_quaternion_count);
    const int common_smpl_joint_count = std::min(source_motion->GetNumSmplJoints(), smpl_joint_count);
    const int common_smpl_pose_count = std::min(source_motion->GetNumSmplPoses(), smpl_pose_count);

    if (source_motion->BodyPartIndexes() != target_motion->BodyPartIndexes()) {
        std::cout << "[MotionTransition] Body layout differs; blending common channels to "
                  << target_motion->name << std::endl;
    }

    const double* source_joint_positions = source_motion->JointPositions(normalized_source_frame);
    const double* source_joint_velocities = source_motion->JointVelocities(normalized_source_frame);
    const auto* source_body_positions = source_motion->BodyPositions(normalized_source_frame);
    const auto* source_body_quaternions = source_motion->BodyQuaternions(normalized_source_frame);
    const auto* source_body_linear_velocities = source_motion->BodyLinVelocities(normalized_source_frame);
    const auto* source_body_angular_velocities = source_motion->BodyAngVelocities(normalized_source_frame);
    const auto* source_smpl_joints = source_motion->SmplJoints(normalized_source_frame);
    const auto* source_smpl_poses = source_motion->SmplPoses(normalized_source_frame);

    const double* target_initial_joint_positions = target_motion->JointPositions(0);
    const double* target_initial_joint_velocities = target_motion->JointVelocities(0);
    const auto* target_initial_body_positions = target_motion->BodyPositions(0);
    const auto* target_initial_body_quaternions = target_motion->BodyQuaternions(0);
    const auto* target_initial_body_linear_velocities = target_motion->BodyLinVelocities(0);
    const auto* target_initial_body_angular_velocities = target_motion->BodyAngVelocities(0);
    const auto* target_initial_smpl_joints = target_motion->SmplJoints(0);
    const auto* target_initial_smpl_poses = target_motion->SmplPoses(0);

    for (int transition_frame = 0; transition_frame < transition_frame_count; ++transition_frame) {
        const double raw_ratio = transition_frame_count <= 1
                                     ? 1.0
                                     : static_cast<double>(transition_frame) /
                                           static_cast<double>(transition_frame_count - 1);
        const double blend_ratio = CalculateSmoothStepRatio(raw_ratio);

        double* destination_joint_positions = transition_motion->JointPositions(transition_frame);
        double* destination_joint_velocities = transition_motion->JointVelocities(transition_frame);
        for (int joint_index = 0; joint_index < joint_count; ++joint_index) {
            if (joint_index < common_joint_count) {
                destination_joint_positions[joint_index] = InterpolateScalar(
                    source_joint_positions[joint_index], target_initial_joint_positions[joint_index], blend_ratio);
                destination_joint_velocities[joint_index] = InterpolateScalar(
                    source_joint_velocities[joint_index], target_initial_joint_velocities[joint_index], blend_ratio);
            } else {
                destination_joint_positions[joint_index] = target_initial_joint_positions[joint_index];
                destination_joint_velocities[joint_index] = target_initial_joint_velocities[joint_index];
            }
        }

        auto* destination_body_positions = transition_motion->BodyPositions(transition_frame);
        auto* destination_body_linear_velocities = transition_motion->BodyLinVelocities(transition_frame);
        auto* destination_body_angular_velocities = transition_motion->BodyAngVelocities(transition_frame);
        for (int body_index = 0; body_index < body_count; ++body_index) {
            for (int axis_index = 0; axis_index < 3; ++axis_index) {
                if (body_index < common_body_count) {
                    destination_body_positions[body_index][axis_index] = InterpolateScalar(
                        source_body_positions[body_index][axis_index],
                        target_initial_body_positions[body_index][axis_index],
                        blend_ratio);
                    destination_body_linear_velocities[body_index][axis_index] = InterpolateScalar(
                        source_body_linear_velocities[body_index][axis_index],
                        target_initial_body_linear_velocities[body_index][axis_index],
                        blend_ratio);
                    destination_body_angular_velocities[body_index][axis_index] = InterpolateScalar(
                        source_body_angular_velocities[body_index][axis_index],
                        target_initial_body_angular_velocities[body_index][axis_index],
                        blend_ratio);
                } else {
                    destination_body_positions[body_index][axis_index] =
                        target_initial_body_positions[body_index][axis_index];
                    destination_body_linear_velocities[body_index][axis_index] =
                        target_initial_body_linear_velocities[body_index][axis_index];
                    destination_body_angular_velocities[body_index][axis_index] =
                        target_initial_body_angular_velocities[body_index][axis_index];
                }
            }
        }

        auto* destination_body_quaternions = transition_motion->BodyQuaternions(transition_frame);
        for (int body_index = 0; body_index < body_quaternion_count; ++body_index) {
            if (body_index < common_body_quaternion_count) {
                destination_body_quaternions[body_index] = quat_slerp_d(
                    source_body_quaternions[body_index], target_initial_body_quaternions[body_index], blend_ratio);
            } else {
                destination_body_quaternions[body_index] = target_initial_body_quaternions[body_index];
            }
        }

        auto* destination_smpl_joints = transition_motion->SmplJoints(transition_frame);
        for (int smpl_joint_index = 0; smpl_joint_index < smpl_joint_count; ++smpl_joint_index) {
            for (int axis_index = 0; axis_index < 3; ++axis_index) {
                if (smpl_joint_index < common_smpl_joint_count) {
                    destination_smpl_joints[smpl_joint_index][axis_index] = InterpolateScalar(
                        source_smpl_joints[smpl_joint_index][axis_index],
                        target_initial_smpl_joints[smpl_joint_index][axis_index],
                        blend_ratio);
                } else {
                    destination_smpl_joints[smpl_joint_index][axis_index] =
                        target_initial_smpl_joints[smpl_joint_index][axis_index];
                }
            }
        }

        auto* destination_smpl_poses = transition_motion->SmplPoses(transition_frame);
        for (int smpl_pose_index = 0; smpl_pose_index < smpl_pose_count; ++smpl_pose_index) {
            for (int axis_index = 0; axis_index < 3; ++axis_index) {
                if (smpl_pose_index < common_smpl_pose_count) {
                    destination_smpl_poses[smpl_pose_index][axis_index] = InterpolateScalar(
                        source_smpl_poses[smpl_pose_index][axis_index],
                        target_initial_smpl_poses[smpl_pose_index][axis_index],
                        blend_ratio);
                } else {
                    destination_smpl_poses[smpl_pose_index][axis_index] =
                        target_initial_smpl_poses[smpl_pose_index][axis_index];
                }
            }
        }
    }

    for (int target_frame = 0; target_frame < appended_target_frame_count; ++target_frame) {
        const int destination_frame = transition_frame_count + target_frame;

        std::copy(target_motion->JointPositions(target_frame),
                  target_motion->JointPositions(target_frame) + joint_count,
                  transition_motion->JointPositions(destination_frame));
        std::copy(target_motion->JointVelocities(target_frame),
                  target_motion->JointVelocities(target_frame) + joint_count,
                  transition_motion->JointVelocities(destination_frame));
        std::copy(target_motion->BodyPositions(target_frame),
                  target_motion->BodyPositions(target_frame) + body_count,
                  transition_motion->BodyPositions(destination_frame));
        std::copy(target_motion->BodyQuaternions(target_frame),
                  target_motion->BodyQuaternions(target_frame) + body_quaternion_count,
                  transition_motion->BodyQuaternions(destination_frame));
        std::copy(target_motion->BodyLinVelocities(target_frame),
                  target_motion->BodyLinVelocities(target_frame) + body_count,
                  transition_motion->BodyLinVelocities(destination_frame));
        std::copy(target_motion->BodyAngVelocities(target_frame),
                  target_motion->BodyAngVelocities(target_frame) + body_count,
                  transition_motion->BodyAngVelocities(destination_frame));
        std::copy(target_motion->SmplJoints(target_frame),
                  target_motion->SmplJoints(target_frame) + smpl_joint_count,
                  transition_motion->SmplJoints(destination_frame));
        std::copy(target_motion->SmplPoses(target_frame),
                  target_motion->SmplPoses(target_frame) + smpl_pose_count,
                  transition_motion->SmplPoses(destination_frame));
    }

    std::cout << "[MotionTransition] Gentle transition to " << target_motion->name
              << " using " << transition_frame_count << " frames" << std::endl;
    return transition_motion;
}

inline bool IsInitialStandingReferenceMotion(
    const std::shared_ptr<const MotionSequence>& motion) {
    return motion && motion->name == "initial_standing_pose";
}

/**
 * @class InputInterface
 * @brief Abstract base class for all input sources (keyboard, gamepad, ZMQ, ROS2).
 *
 * Concrete sub-classes must implement update() and handle_input().
 * The base class owns shared state (VR buffers, compliance values, stdin queue)
 * that all implementations can use.
 */
class InputInterface {
public:

    /// Identifies the physical / logical source of this input interface.
    enum class InputType {
      KEYBOARD,   ///< Terminal stdin keyboard
      GAMEPAD,    ///< Unitree wireless gamepad (raw joystick data)
      ROS2,       ///< ROS 2 topics (msgpack-serialised ControlGoalMsg)
      NETWORK,    ///< ZMQ packed-message protocol (pose / planner topics)
      UNKNOWN     ///< Default / uninitialised
    };

    /// Virtual destructor ensures correct cleanup of derived classes.
    virtual ~InputInterface() = default;

    // ------------------------------------------------------------------
    // Pure-virtual interface
    // ------------------------------------------------------------------

    /**
     * @brief Poll the input source for new data and latch it internally.
     *
     * Called once per control-loop tick **before** handle_input().
     * Implementations should:
     *  - Read all available bytes / messages from their source.
     *  - Update per-frame boolean flags (e.g. start_control, delta_left).
     *  - NOT modify shared system state – that is done in handle_input().
     */
    virtual void update() = 0;

    /**
     * @brief Apply latched input flags to the system state.
     *
     * Called once per control-loop tick **after** update().
     *
     * @param motion_reader         Reference to the pre-loaded motion library.
     * @param current_motion        Currently-active MotionSequence (may be swapped).
     * @param current_frame         Playback cursor within current_motion.
     * @param operator_state        High-level operator signals (start/stop/play).
     * @param reinitialize_heading  Set to true to recapture the IMU heading.
     * @param heading_state_buffer  Thread-safe heading buffer (delta heading, init quat).
     * @param has_planner           Whether a locomotion planner is loaded.
     * @param planner_state         Planner enable / initialise state.
     * @param movement_state_buffer Thread-safe buffer for locomotion commands.
     * @param current_motion_mutex  Guards concurrent access to current_motion/frame.
     */
    virtual void handle_input(MotionDataReader& motion_reader,
                            std::shared_ptr<const MotionSequence>& current_motion,
                            int& current_frame,
                            OperatorState& operator_state,
                            bool& reinitialize_heading,
                            DataBuffer<HeadingState>& heading_state_buffer,
                            bool has_planner,
                            PlannerState& planner_state,
                            DataBuffer<MovementState>& movement_state_buffer,
                            std::mutex& current_motion_mutex,
                            bool& report_temperature) = 0;

    // ------------------------------------------------------------------
    // Capability queries – overridden by sub-classes as needed
    // ------------------------------------------------------------------

    /// @return The InputType tag for this concrete implementation.
    virtual InputType GetType() {
      return type_;
    }

    /// @return True if this interface provides upper-body joint targets (17 DOF).
    virtual bool HasUpperBodyControl() const {
      return has_upper_body_control_;
    }
    
    /// @return True if this interface provides VR 3-point tracking data
    ///         (left wrist, right wrist, head – 9 position + 12 orientation values).
    virtual bool HasVR3PointControl() const {
      return has_vr_3point_control_;
    }
    
    /// @return True if this interface provides VR 5-point tracking data
    ///         (3-point + 2 extra trackers – 15 position + 20 orientation values).
    virtual bool HasVR5PointControl() const {
      return has_vr_5point_control_;
    }

    /// @return True if this interface provides Dex3 hand joint targets (7 DOF per hand).
    virtual bool HasHandJoints() const {
      return has_hand_joints_;
    }
    
    /// @return True if an external token-state vector is available (e.g. from ROS2/ZMQ).
    virtual bool HasExternalTokenState() const {
      return has_external_token_state_;
    }
    
    /**
     * @brief Retrieve the latest external token-state vector (if any).
     * @return A pair: {true, vector} if data is available; {false, {}} otherwise.
     */
    virtual std::pair<bool, std::vector<double>> GetExternalTokenState() const {
      if (!has_external_token_state_) {
        return {false, {}};
      }
      auto buffered_data = external_token_state_.GetDataWithTime();
      if (buffered_data.data) {
        return {true, *buffered_data.data};
      }
      return {false, {}};
    }
    
    // ------------------------------------------------------------------
    // VR 3-point tracking data accessors
    // Layout: [left_wrist(xyz), right_wrist(xyz), head(xyz)]
    //         [left_quat(wxyz), right_quat(wxyz), head_quat(wxyz)]
    // ------------------------------------------------------------------

    /// Write new VR 3-point position data into the thread-safe buffer.
    virtual void SetVR3PointPosition(const std::array<double, 9>& position) {
        vr_3point_position_.SetData(position);
    }
    
    /// Write new VR 3-point orientation data into the thread-safe buffer.
    virtual void SetVR3PointOrientation(const std::array<double, 12>& orientation) {
        vr_3point_orientation_.SetData(orientation);
    }

    /**
     * @brief Get the latest VR 3-point position data.
     * @return {true, positions} if VR tracking is active and data is available;
     *         {false, default_positions} otherwise.
     *
     * Default positions (metres, in body frame):
     *   left wrist  = ( 0.0903,  0.1615, -0.2411)
     *   right wrist = ( 0.1280, -0.1522, -0.2461)
     *   head        = ( 0.0241, -0.0081,  0.4028)
     */
    virtual std::pair<bool, std::array<double, 9>> GetVR3PointPosition() const {
        if(!has_vr_3point_control_) {
            return {false, {0.0903,  0.1615, -0.2411, 
                    0.1280, -0.1522, -0.2461, 
                    0.0241, -0.0081,  0.4028}};
        }
        auto buffered_data = vr_3point_position_.GetDataWithTime();
        if (buffered_data.data) {
            return {true, *buffered_data.data};
        }
        return {false, {0.0903,  0.1615, -0.2411, 
                0.1280, -0.1522, -0.2461, 
                0.0241, -0.0081,  0.4028}};
    }
    
    /**
     * @brief Get the latest VR 3-point orientation data (quaternions w,x,y,z).
     * @return {true, orientations} if VR tracking is active and data is available;
     *         {false, default_orientations} otherwise.
     */
    virtual std::pair<bool, std::array<double, 12>> GetVR3PointOrientation() const {
        if(!has_vr_3point_control_) {
            return {false, {0.7295,  0.3145,  0.5533, -0.2506, 
                    0.7320, -0.2639,  0.5395,  0.3217, 
                    0.9991,  0.011,  0.0402, -0.0002}};
        }
        auto buffered_data = vr_3point_orientation_.GetDataWithTime();
        if (buffered_data.data) {
            return {true, *buffered_data.data};
        }
        return {false, {0.7295,  0.3145,  0.5533, -0.2506, 
                0.7320, -0.2639,  0.5395,  0.3217, 
                0.9991,  0.011,  0.0402, -0.0002}};
    }
    
    /**
     * @brief Get VR 3-point compliance values (keyboard-controlled).
     *
     * Compliance is always controlled via keyboard (g/h keys for left, b/v for right)
     * and is intentionally **not** overwritten by external sources (ROS2 / ZMQ).
     * @return [left_arm_compliance, right_arm_compliance, head_compliance] in [0, 0.5].
     */
    virtual std::array<double, 3> GetVR3PointCompliance() const {
        auto buffered_data = vr_3point_compliance_.GetDataWithTime();
        if (buffered_data.data) {
            return *buffered_data.data;
        }
        return {0.5, 0.5, 0.0};
    }

    /// Set initial VR 3-point compliance values (called at startup from command-line args).
    virtual void SetVR3PointCompliance(const std::array<double, 3>& compliance) {
        vr_3point_compliance_.SetData(compliance);
    }
    
    /// Adjust left-arm (index 0) compliance by @p delta, clamped to [0.0, 0.5].
    /// Only takes effect if the active policy observes 'vr_3point_compliance'.
    virtual void AdjustLeftHandCompliance(double delta) {
        auto compliance = GetVR3PointCompliance();
        compliance[0] = std::clamp(compliance[0] + delta, 0.0, 0.5);
        SetVR3PointCompliance(compliance);
        std::cout << "[Compliance] Left hand: " <<  GetVR3PointCompliance()[0] << std::endl;
    }
    
    /// Adjust right-arm (index 1) compliance by @p delta, clamped to [0.0, 0.5].
    /// Only takes effect if the active policy observes 'vr_3point_compliance'.
    virtual void AdjustRightHandCompliance(double delta) {
        auto compliance = GetVR3PointCompliance();
        compliance[1] = std::clamp(compliance[1] + delta, 0.0, 0.5);
        SetVR3PointCompliance(compliance);
        std::cout << "[Compliance] Right hand: " <<  GetVR3PointCompliance()[1] << std::endl;
    }

    // =========================================================================
    // Hand max close ratio control (keyboard-controlled via X/C keys)
    // Controls how much the Dex3 hands can close (0.2 = 80% open, 1.0 = fully closed)
    // =========================================================================
    
    // Get the current max close ratio (keyboard-controlled)
    virtual double GetMaxCloseRatio() const {
        return max_close_ratio_;
    }
    
    // Set initial max close ratio (from command line or initialization)
    virtual void SetMaxCloseRatio(double ratio) {
        max_close_ratio_.store(std::clamp(ratio, 0.2, 1.0), std::memory_order_relaxed);
    }
    
    // Adjust max close ratio by delta (X = +0.1, C = -0.1), clipped to [0.2, 1.0]
    virtual void AdjustMaxCloseRatio(double delta) {
        double new_val = std::clamp(max_close_ratio_.load(std::memory_order_relaxed) + delta, 0.2, 1.0);
        max_close_ratio_.store(new_val, std::memory_order_relaxed);
        std::cout << "[InputInterface] Max close ratio adjusted to: " << new_val 
                  << " (range: 0.2-1.0, higher = more closed)" << std::endl;
    }

    // ------------------------------------------------------------------
    // VR 5-point tracking (3-point + 2 extra body trackers)
    // Layout: [left_wrist, right_wrist, head, tracker1, tracker2] × xyz
    // ------------------------------------------------------------------

    /**
     * @brief Get VR 5-point position data (15 doubles).
     * @return {true, positions} if 5-point tracking is active; {false, defaults} otherwise.
     */
    virtual std::pair<bool, std::array<double, 15>> GetVR5PointPosition() const {
        if(!has_vr_5point_control_) {
            return {false, {0.0903,  0.1615, -0.2411, 
                    0.1280, -0.1522, -0.2461, 
                    0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0}};
        }
        auto buffered_data = vr_5point_position_.GetDataWithTime();
        if (buffered_data.data) {
            return {true, *buffered_data.data};
        }
        return {false, {0.0903,  0.1615, -0.2411, 
                0.1280, -0.1522, -0.2461, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0}};
    }

    /// @brief Get VR 5-point orientation data (20 doubles = 5 quaternions × wxyz).
    virtual std::pair<bool, std::array<double, 20>> GetVR5PointOrientation() const {
        if(!has_vr_5point_control_) {
            return {false, {0.7295,  0.3145,  0.5533, -0.2506, 
                    0.7320, -0.2639,  0.5395,  0.3217, 
                    0.9991,  0.011,  0.0402, -0.0002, 
                    1.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0}};
        }
        auto buffered_data = vr_5point_orientation_.GetDataWithTime();
        if (buffered_data.data) {
            return {true, *buffered_data.data};
        }
        return {false, {0.7295,  0.3145,  0.5533, -0.2506, 
                0.7320, -0.2639,  0.5395,  0.3217, 
                0.9991,  0.011,  0.0402, -0.0002, 
                1.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0}};
    }

    // ------------------------------------------------------------------
    // Hand / upper-body joint accessors
    // ------------------------------------------------------------------

    /**
     * @brief Get 7-DOF Dex3 hand joint positions.
     * @param is_left  true → left hand, false → right hand.
     * @return {true, joints} if hand joint data is available; {false, defaults} otherwise.
     *         Default left  = {0, 0,  1.75, -1.57, -1.75, -1.57, -1.75}
     *         Default right = {0, 0, -1.75,  1.57,  1.75,  1.57,  1.75}
     */
    virtual std::pair<bool, std::array<double, 7>> GetHandPose(bool is_left) const {
        if(!has_hand_joints_) {
            if(is_left) {
                return {false, {0, 0, 1.75, -1.57, -1.75, -1.57, -1.75 }};
            } else {
                return {false, {0, 0, -1.75,  1.57,  1.75,  1.57,  1.75 }};
            }
        }
        if(is_left) {
            auto buffered_data = left_hand_joint_.GetDataWithTime();
            if (buffered_data.data) {
                return {true, *buffered_data.data};
            }
            else {
                return {false, {0, 0, 1.75, -1.57, -1.75, -1.57, -1.75 }};
            }
        } else {
            auto buffered_data = right_hand_joint_.GetDataWithTime();
            if (buffered_data.data) {
                return {true, *buffered_data.data};
            } else {
                return {false, {0, 0, -1.75,  1.57,  1.75,  1.57,  1.75 }};
            }
        }
    }

    /// @brief Get upper-body joint target positions (17 DOF, radians).
    /// @return {true, positions} if upper-body data is available; {false, zeros} otherwise.
    virtual std::pair<bool, std::array<double, 17>> GetUpperBodyJointPositions() const {
        if(!has_upper_body_control_) {
            return {false, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        }
        auto buffered_data = upper_body_joint_positions_.GetDataWithTime();
        if (buffered_data.data) {
            return {true, *buffered_data.data};
        }
        return {false, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    }

    /// @brief Get upper-body joint target velocities (17 DOF, rad/s).
    /// @return {true, velocities} if upper-body data is available; {false, zeros} otherwise.
    virtual std::pair<bool, std::array<double, 17>> GetUpperBodyJointVelocities() const {
        if(!has_upper_body_control_) {
            return {false, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        }
        auto buffered_data = upper_body_joint_velocities_.GetDataWithTime();
        if (buffered_data.data) {
            return {true, *buffered_data.data};
        }
        return {false, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    }

    /// @brief Optional planner-mode state exposed by local input handlers.
    virtual std::optional<bool> GetPlannerModeEnabled() const {
        return {};
    }

    /// @brief Optional planner locomotion mode selected by the input handler.
    virtual std::optional<int> GetPlannerLocomotionMode() const {
        return {};
    }

    /// @brief Optional planner movement speed selected by the input handler.
    virtual std::optional<double> GetPlannerMovementSpeed() const {
        return {};
    }

    /// @brief Optional planner body height selected by the input handler.
    virtual std::optional<double> GetPlannerHeight() const {
        return {};
    }

    /// @brief Get the last update time of the input interface.
    virtual std::optional<std::chrono::steady_clock::time_point> GetLastUpdateTime() const {
        // Return empty optional as a default, implementers should override this if they have a timestamp
        return {};
    }

    /// @brief Consume a one-shot request to reload the reference motion library.
    virtual bool ConsumeReloadMotionsRequest() {
        return false;
    }

    // ------------------------------------------------------------------
    // Shared stdin buffering
    // ------------------------------------------------------------------
    /**
     * @brief Read one character, trying the manager-pushed buffer first, then raw stdin.
     *
     * When an InterfaceManager is active, it pre-reads stdin and dispatches
     * keys: manager-level shortcuts are consumed, everything else is queued
     * via PushStdinChar() for the active interface to read later.
     *
     * @param[out] ch  The character that was read.
     * @return True if a character was available, false otherwise.
     */
    bool ReadStdinChar(char& ch) {
        // First check if we have buffered keys from manager
        if (!stdin_buffer_.empty()) {
            ch = stdin_buffer_.front();
            stdin_buffer_.pop();
            return true;
        }
        // Otherwise read directly from stdin (non-blocking)
        return read(STDIN_FILENO, &ch, 1) > 0;
    }

    /// Queue a key into this interface's stdin buffer (used by managers).
    void PushStdinChar(char ch) {
        stdin_buffer_.push(ch);
    }

    // ------------------------------------------------------------------
    // Safety-reset mechanism  (used when switching active interfaces)
    // ------------------------------------------------------------------

    /// Flag that tells the interface to abandon its current mode and return
    /// to a safe default (disable planner, stop streaming, etc.).
    std::atomic<bool> reset_to_safe_state_{false};

    /// Called by the manager to request a safety reset on this interface.
    void TriggerSafetyReset() {
        reset_to_safe_state_.store(true, std::memory_order_release);
    }

    /// Atomically check and clear the safety-reset flag.
    /// @return True if a reset was requested since the last call.
    bool CheckAndClearSafetyReset() {
        return reset_to_safe_state_.exchange(false, std::memory_order_acq_rel);
    }

protected:
    /// Protected default constructor – only concrete sub-classes may instantiate.
    InputInterface() = default;

    // Non-copyable, non-movable (instances are managed via unique_ptr).
    InputInterface(const InputInterface&) = delete;
    InputInterface& operator=(const InputInterface&) = delete;
    InputInterface(InputInterface&&) = delete;
    InputInterface& operator=(InputInterface&&) = delete;

    // ------------------------------------------------------------------
    // Capability flags – set by sub-class constructors / update methods
    // ------------------------------------------------------------------
    InputType type_ = InputType::UNKNOWN;             ///< Concrete input source tag.
    std::atomic<bool> has_vr_3point_control_{false};  ///< VR 3-point tracking available.
    std::atomic<bool> has_vr_5point_control_{false};  ///< VR 5-point tracking available.
    std::atomic<bool> has_hand_joints_{false};        ///< Dex3 hand joint data available.
    std::atomic<bool> has_external_token_state_{false}; ///< External token-state vector available.
    std::atomic<bool> has_upper_body_control_{false}; ///< Upper-body 17-DOF targets available.

    // ------------------------------------------------------------------
    // Thread-safe data buffers (written by input threads, read by control loop)
    // ------------------------------------------------------------------

    /// VR 3-point positions [left_wrist xyz, right_wrist xyz, head xyz].
    DataBuffer<std::array<double, 9>> vr_3point_position_;
    /// VR 3-point orientations [left quat wxyz, right quat wxyz, head quat wxyz].
    DataBuffer<std::array<double, 12>> vr_3point_orientation_;
    /// VR 3-point compliance [left_arm, right_arm, head] – keyboard-controlled.
    DataBuffer<std::array<double, 3>> vr_3point_compliance_;
    /// VR 5-point positions (3-point + 2 extra trackers × xyz = 15 values).
    DataBuffer<std::array<double, 15>> vr_5point_position_;
    /// VR 5-point orientations (5 quaternions × wxyz = 20 values).
    DataBuffer<std::array<double, 20>> vr_5point_orientation_;

    /// Upper-body target joint positions (17 DOF, radians).
    DataBuffer<std::array<double, 17>> upper_body_joint_positions_;
    /// Upper-body target joint velocities (17 DOF, rad/s).
    DataBuffer<std::array<double, 17>> upper_body_joint_velocities_;
    
    /// Left-hand Dex3 joint positions (7 DOF).
    DataBuffer<std::array<double, 7>> left_hand_joint_;
    /// Right-hand Dex3 joint positions (7 DOF).
    DataBuffer<std::array<double, 7>> right_hand_joint_;
    
    /// Arbitrary external token-state vector (e.g. latent codes from a remote model).
    DataBuffer<std::vector<double>> external_token_state_;

    /// Keyboard-controlled max close ratio for Dex3 hands.
    /// Adjusted via keyboard (X = +0.1, C = −0.1), clamped to [0.2, 1.0].
    /// 1.0 = fully closed allowed (default); use --max-close-ratio CLI arg to limit.
    std::atomic<double> max_close_ratio_{1.0};

    /// Shared stdin buffer – the InterfaceManager pushes non-manager keys here
    /// for the currently-active interface to consume via ReadStdinChar().
    std::queue<char> stdin_buffer_;
};

#endif // INPUT_INTERFACE_HPP
