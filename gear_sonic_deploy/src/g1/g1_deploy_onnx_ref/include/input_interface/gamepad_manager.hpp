/**
 * @file gamepad_manager.hpp
 * @brief Gamepad manager with official motion/planner button mapping.
 *
 * GamepadManager is a specialised InputInterface that:
 *  - Reads the Unitree wireless gamepad **directly** (no nested Gamepad class).
 *  - Uses F1 to toggle between reference-motion playback and planner mode.
 *  - Keeps ZMQ available through keyboard manager shortcuts, not gamepad F1.
 *
 * ## Gamepad Button Mapping
 *
 *   Button  | Action
 *   --------|-------
 *   Start   | Start control
 *   Select  | Emergency stop
 *   F1      | Toggle motion mode / planner mode
 *   X/Y     | Reinitialize base heading
 *   D-left/right | Delta heading in motion mode
 *
 *   Motion mode:
 *   A       | Play / resume current reference motion
 *   B       | Restart current reference motion at frame 0
 *   L1/R1   | Previous / next reference motion
 *
 *   Planner mode:
 *   A       | Play / resume
 *   B       | Planner emergency stop
 *   L1/R1   | Previous / next planner locomotion mode
 *   L2/R2   | Decrease / increase planner speed or height
 *   L stick | Movement direction
 *   R stick | Facing direction
 */

#ifndef GAMEPAD_MANAGER_HPP
#define GAMEPAD_MANAGER_HPP

#include <memory>
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <array>
#include <thread>
#include <chrono>
#include <algorithm>

#include "input_interface.hpp"
#include "zmq_endpoint_interface.hpp"
#include "gamepad.hpp"
#include "../localmotion_kplanner.hpp"  // For LocomotionMode enum

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @class GamepadManager
 * @brief Gamepad controller with official motion/planner button mapping.
 *
 * When in GAMEPAD mode, F1 toggles between reference-motion playback and
 * planner control. ZMQ remains available via keyboard manager shortcuts.
 */
class GamepadManager : public InputInterface {
  public:
    // ========================================
    // DEBUG CONTROL FLAG
    // ========================================
    static constexpr bool DEBUG_LOGGING = true;

    enum class ManagedType {
      GAMEPAD = 0,
      ZMQ = 1
    };

    GamepadManager(
      const std::string& zmq_host,
      int zmq_port,
      const std::string& zmq_topic,
      bool zmq_conflate,
      bool zmq_verbose
    ) : InputInterface(), zmq_host_(zmq_host), zmq_port_(zmq_port), zmq_topic_(zmq_topic),
        zmq_conflate_(zmq_conflate), zmq_verbose_(zmq_verbose) {
      type_ = InputType::GAMEPAD;  // Default to gamepad mode
      buildInterfaces();
      active_ = ManagedType::GAMEPAD;
      current_ = nullptr;  // Gamepad mode doesn't use delegate

      // Initialize with Standing motion set, SLOW_WALK as default
      motion_set_index_ = 0;
      current_motion_set_ = get_motion_set(motion_set_index_);
      mode_index_in_set_ = 0;
      applyModeFromSet();
    }

    void update() override {
      // Reset per-frame flags
      emergency_stop_ = false;
      report_temperature_flag_ = false;
      start_control_ = false;
      stop_control_ = false;
      motion_prev_ = false;
      motion_next_ = false;
      play_motion_ = false;
      motion_restart_ = false;
      delta_left_ = false;
      delta_right_ = false;
      reinitialize_ = false;
      planner_emergency_stop_ = false;

      // Handle stdin shortcuts for switching and emergency stop
      char ch;
      while (ReadStdinChar(ch)) {
        bool is_manager_key = false;
        switch (ch) {
          case '@':
            SetActiveInterface(ManagedType::GAMEPAD);
            is_manager_key = true;
            break;
          case '#':
            SetActiveInterface(ManagedType::ZMQ);
            is_manager_key = true;
            break;
          case 'o':
          case 'O':
            emergency_stop_ = true;
            is_manager_key = true;
            std::cout << "[GamepadManager] EMERGENCY STOP triggered (O/o key pressed)" << std::endl;
            break;
          case 'f':
          case 'F':
            report_temperature_flag_ = true;
            is_manager_key = true;
            break;
        }

        if (!is_manager_key && current_) {
          current_->PushStdinChar(ch);
        }
      }

      // Update gamepad data and buttons
      update_gamepad_data(gamepad_data_.RF_RX);

      // F1 - Toggle between reference-motion mode and planner mode.
      if (F1_.on_press) {
        if (active_ != ManagedType::GAMEPAD) {
          SetActiveInterface(ManagedType::GAMEPAD);
        }
        use_planner_ = !use_planner_;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] F1 pressed - Planner toggled to: "
                    << (use_planner_ ? "ON" : "OFF") << std::endl;
        }
      }

      // Select - Emergency Stop
      if (select_.on_press) {
        stop_control_ = true;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Select pressed - Emergency Stop" << std::endl;
        }
      }

      // If not in gamepad mode, update the active interface
      if (active_ != ManagedType::GAMEPAD && current_) {
        current_->update();
      } else {
        processGamepadPlannerControls();
      }
    }

    void handle_input(MotionDataReader& motion_reader,
                      std::shared_ptr<const MotionSequence>& current_motion,
                      int& current_frame,
                      OperatorState& operator_state,
                      bool& reinitialize_heading,
                      DataBuffer<HeadingState>& heading_state_buffer,
                      bool has_planner,
                      PlannerState& planner_state,
                      DataBuffer<MovementState>& movement_state_buffer,
                      std::mutex& current_motion_mutex,
                      bool& report_temperature) override {
      if (use_planner_ && !has_planner) {
        std::cerr << "[GamepadManager ERROR] Planner not loaded - cannot enable planner mode." << std::endl;
        use_planner_ = false;
      }

      // NOTE: Encoder mode safety check removed - encoder mode is now a property of the motion
      // If a motion has an incompatible encoder mode for gamepad mode, the user should
      // switch to a different motion with a compatible encoder mode

      // Global emergency stop
      if (emergency_stop_) {
        operator_state.stop = true;
      }

      // Global temperature report (F key)
      if (report_temperature_flag_) {
        report_temperature = true;
        report_temperature_flag_ = false;
      }

      // Handle stop control
      if (stop_control_) { operator_state.stop = true; }

      // D-pad left/right are common heading nudges in both motion and planner modes.
      if (delta_left_) {
        auto current_heading_state = heading_state_buffer.GetDataWithTime().data;
        HeadingState current_state = current_heading_state ? *current_heading_state : HeadingState();
        double new_delta = current_state.delta_heading + 0.1;
        heading_state_buffer.SetData(HeadingState(current_state.init_base_quat, new_delta));
        std::cout << "[GamepadManager] Delta heading left: " << new_delta << " rad" << std::endl;
      }

      if (delta_right_) {
        auto current_heading_state = heading_state_buffer.GetDataWithTime().data;
        HeadingState current_state = current_heading_state ? *current_heading_state : HeadingState();
        double new_delta = current_state.delta_heading - 0.1;
        heading_state_buffer.SetData(HeadingState(current_state.init_base_quat, new_delta));
        std::cout << "[GamepadManager] Delta heading right: " << new_delta << " rad" << std::endl;
      }

      // If in gamepad mode, handle official motion/planner controls
      if (active_ == ManagedType::GAMEPAD) {
        handlePlannerToggle(motion_reader, current_motion, current_frame,
                            operator_state, reinitialize_heading, has_planner,
                            planner_state, movement_state_buffer, current_motion_mutex);
        if (use_planner_) {
          handleGamepadPlannerInput(motion_reader, current_motion, current_frame,
                                    operator_state, reinitialize_heading, heading_state_buffer,
                                    planner_state, movement_state_buffer, current_motion_mutex);
        } else {
          handleGamepadMotionInput(motion_reader, current_motion, current_frame,
                                   operator_state, reinitialize_heading, heading_state_buffer,
                                   current_motion_mutex);
        }
      } else {
        // Delegate to ZMQ
        if (current_) {
          current_->handle_input(motion_reader, current_motion, current_frame, operator_state,
                                reinitialize_heading, heading_state_buffer, has_planner,
                                planner_state, movement_state_buffer, current_motion_mutex, report_temperature);
        }
      }
    }

    bool HasVR3PointControl() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->HasVR3PointControl();
      }
      return has_vr_3point_control_;
    }

    bool HasHandJoints() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->HasHandJoints();
      }
      return has_hand_joints_;
    }

    bool HasExternalTokenState() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->HasExternalTokenState();
      }
      return has_external_token_state_;
    }

    std::pair<bool, std::array<double, 9>> GetVR3PointPosition() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->GetVR3PointPosition();
      }
      return InputInterface::GetVR3PointPosition();
    }

    std::pair<bool, std::array<double, 12>> GetVR3PointOrientation() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->GetVR3PointOrientation();
      }
      return InputInterface::GetVR3PointOrientation();
    }

    std::array<double, 3> GetVR3PointCompliance() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->GetVR3PointCompliance();
      }
      return InputInterface::GetVR3PointCompliance();
    }

    std::pair<bool, std::array<double, 7>> GetHandPose(bool is_left) const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->GetHandPose(is_left);
      }
      return InputInterface::GetHandPose(is_left);
    }

    std::pair<bool, std::vector<double>> GetExternalTokenState() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->GetExternalTokenState();
      }
      return InputInterface::GetExternalTokenState();
    }

    // Receive raw wireless remote data for gamepad
    void UpdateGamepadRemoteData(const uint8_t* buff, size_t size) {
      if (buff == nullptr || size == 0) { return; }
      size_t copy_size = std::min<size_t>(size, sizeof(gamepad_data_.buff));
      std::memcpy(gamepad_data_.buff, buff, copy_size);
    }

    void SetActiveInterface(ManagedType t) {
      for (size_t i = 0; i < order_.size(); ++i) {
        if (order_[i] == t) {
          setActiveIndex(static_cast<int>(i));
          return;
        }
      }
    }

    ManagedType GetActiveInterface() const { return active_; }

  private:
    void buildInterfaces() {
      order_.push_back(ManagedType::GAMEPAD);

      zmq_ = std::make_unique<ZMQEndpointInterface>(
        zmq_host_, zmq_port_, zmq_topic_, zmq_conflate_, zmq_verbose_
      );
      order_.push_back(ManagedType::ZMQ);
    }

    void setActiveIndex(int idx) {
      if (order_.empty()) { return; }
      if (idx < 0) { idx = static_cast<int>(order_.size()) - 1; }
      if (idx >= static_cast<int>(order_.size())) { idx = 0; }

      // Trigger safety reset on all managed interfaces when switching
      TriggerSafetyReset();  // Self (for gamepad mode)
      if (zmq_) zmq_->TriggerSafetyReset();

      active_index_ = idx;
      active_ = order_[static_cast<size_t>(active_index_)];

      switch (active_) {
        case ManagedType::GAMEPAD:
          current_ = nullptr;  // Gamepad mode is handled directly
          type_ = InputType::GAMEPAD;
          std::cout << "[GamepadManager] Switched to: GAMEPAD (safety reset triggered)" << std::endl;
          break;
        case ManagedType::ZMQ:
          current_ = zmq_.get();
          type_ = InputType::NETWORK;
          std::cout << "[GamepadManager] Switched to: ZMQ (safety reset triggered)" << std::endl;
          break;
      }
    }

    // Update gamepad analog and button data
    void update_gamepad_data(unitree::common::xRockerBtnDataStruct& key_data) {
      lx_ = lx_ * (1 - smooth_) + (std::fabs(key_data.lx) < dead_zone_ ? 0.0f : key_data.lx) * smooth_;
      rx_ = rx_ * (1 - smooth_) + (std::fabs(key_data.rx) < dead_zone_ ? 0.0f : key_data.rx) * smooth_;
      ry_ = ry_ * (1 - smooth_) + (std::fabs(key_data.ry) < dead_zone_ ? 0.0f : key_data.ry) * smooth_;
      l2_ = l2_ * (1 - smooth_) + (std::fabs(key_data.L2) < dead_zone_ ? 0.0f : key_data.L2) * smooth_;
      ly_ = ly_ * (1 - smooth_) + (std::fabs(key_data.ly) < dead_zone_ ? 0.0f : key_data.ly) * smooth_;

      R1_.update(key_data.btn.components.R1);
      L1_.update(key_data.btn.components.L1);
      start_.update(key_data.btn.components.start);
      select_.update(key_data.btn.components.select);
      R2_.update(key_data.btn.components.R2);
      L2_.update(key_data.btn.components.L2);
      F1_.update(key_data.btn.components.F1);
      F2_.update(key_data.btn.components.F2);
      A_.update(key_data.btn.components.A);
      B_.update(key_data.btn.components.B);
      X_.update(key_data.btn.components.X);
      Y_.update(key_data.btn.components.Y);
      up_.update(key_data.btn.components.up);
      right_.update(key_data.btn.components.right);
      down_.update(key_data.btn.components.down);
      left_.update(key_data.btn.components.left);
    }

    // Select a motion set by index and reset to its default mode
    void selectMotionSet(int set_index) {
      bool was_crawling = isInCrawlingMode();

      motion_set_index_ = set_index;
      current_motion_set_ = get_motion_set(motion_set_index_);
      mode_index_in_set_ = 0;
      boxing_revert_time_ = {};  // Clear boxing timer on set switch

      if (was_crawling) {
        // Soft exit: crawl → kneel → target (staged)
        LocomotionMode target = current_motion_set_[0];
        beginExitCrawlingTransition(target);
      } else {
        // Clear any pending transition and apply directly
        transition_target_mode_ = LocomotionMode::IDLE;
        transition_final_mode_ = LocomotionMode::IDLE;
        transition_start_time_ = {};
        applyModeFromSet();
      }

      // Boxing set enters with IDEL_BOXING → auto-revert to WALK_BOXING after 1s
      if (motion_set_index_ == 2) {
        boxing_revert_time_ = std::chrono::steady_clock::now();
      }
      if constexpr (DEBUG_LOGGING) {
        std::cout << "[GamepadManager DEBUG] D-pad - Motion set " << motion_set_index_
                  << ", mode: " << planner_use_movement_mode_ << std::endl;
      }
    }

    // Apply mode/speed/height from the current mode_index_in_set_
    void applyModeFromSet() {
      if (current_motion_set_.empty()) return;
      LocomotionMode mode = current_motion_set_[mode_index_in_set_];
      planner_use_movement_mode_ = static_cast<int>(mode);
      applySpeedAndHeight(mode);
    }

    // Set fixed speed and height for a given mode
    void applySpeedAndHeight(LocomotionMode mode) {
      switch (mode) {
        case LocomotionMode::SLOW_WALK:
          planner_use_movement_speed_ = 0.4;
          planner_use_height_ = -1.0;
          break;
        case LocomotionMode::RUN:
          planner_use_movement_speed_ = 1.5;
          planner_use_height_ = -1.0;
          break;
        case LocomotionMode::CRAWLING:
          planner_use_movement_speed_ = 0.7;
          planner_use_height_ = 0.4;
          break;
        case LocomotionMode::ELBOW_CRAWLING:
          planner_use_movement_speed_ = 0.7;
          planner_use_height_ = 0.3;
          break;
        case LocomotionMode::IDEL_SQUAT:
        case LocomotionMode::IDEL_KNEEL_TWO_LEGS:
        case LocomotionMode::IDEL_KNEEL:
          planner_use_movement_speed_ = -1.0;
          planner_use_height_ = 0.4;
          break;
        case LocomotionMode::IDEL_BOXING:
        case LocomotionMode::WALK_BOXING:
        case LocomotionMode::LEFT_PUNCH:
        case LocomotionMode::RIGHT_PUNCH:
        case LocomotionMode::RANDOM_PUNCH:
        case LocomotionMode::LEFT_HOOK:
        case LocomotionMode::RIGHT_HOOK:
          planner_use_movement_speed_ = 0.7;
          planner_use_height_ = -1.0;
          break;
        default:
          // WALK, styled walking, and other modes use defaults
          planner_use_movement_speed_ = -1.0;
          planner_use_height_ = -1.0;
          break;
      }
    }

    // Begin a staged crawling transition to target_mode, skipping completed stages
    void beginCrawlingTransition(LocomotionMode target_mode) {
      LocomotionMode current = static_cast<LocomotionMode>(planner_use_movement_mode_);

      if (target_mode == LocomotionMode::CRAWLING) {
        if (current == LocomotionMode::CRAWLING) return; // Already there
        if (current == LocomotionMode::IDEL_KNEEL_TWO_LEGS ||
            current == LocomotionMode::IDEL_KNEEL) {
          // Already kneeling, transition directly to crawling after delay
          transition_target_mode_ = LocomotionMode::CRAWLING;
          transition_start_time_ = std::chrono::steady_clock::now();
        } else {
          // Need to kneel first
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
          applySpeedAndHeight(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
          transition_target_mode_ = LocomotionMode::CRAWLING;
          transition_start_time_ = std::chrono::steady_clock::now();
        }
      } else if (target_mode == LocomotionMode::ELBOW_CRAWLING) {
        if (current == LocomotionMode::ELBOW_CRAWLING) return; // Already there
        if (current == LocomotionMode::CRAWLING) {
          // Already crawling, transition to elbow crawling after delay
          transition_target_mode_ = LocomotionMode::ELBOW_CRAWLING;
          transition_start_time_ = std::chrono::steady_clock::now();
        } else if (current == LocomotionMode::IDEL_KNEEL_TWO_LEGS ||
                   current == LocomotionMode::IDEL_KNEEL) {
          // Already kneeling, go to crawling first then elbow
          transition_target_mode_ = LocomotionMode::CRAWLING;
          transition_final_mode_ = LocomotionMode::ELBOW_CRAWLING;
          transition_start_time_ = std::chrono::steady_clock::now();
        } else {
          // Need to kneel first, then crawl, then elbow
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
          applySpeedAndHeight(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
          transition_target_mode_ = LocomotionMode::CRAWLING;
          transition_final_mode_ = LocomotionMode::ELBOW_CRAWLING;
          transition_start_time_ = std::chrono::steady_clock::now();
        }
      }
      if constexpr (DEBUG_LOGGING) {
        std::cout << "[GamepadManager DEBUG] Crawling transition started -> target: "
                  << static_cast<int>(target_mode) << std::endl;
      }
    }

    // Begin a staged exit from crawling/elbow crawling to target_mode (reverse of enter)
    // ELBOW_CRAWLING → (immediate CRAWLING) → (2s) KNEEL → (2s) target
    // CRAWLING → (immediate KNEEL) → (2s) target
    void beginExitCrawlingTransition(LocomotionMode target_mode) {
      LocomotionMode current = static_cast<LocomotionMode>(planner_use_movement_mode_);

      if (current == LocomotionMode::ELBOW_CRAWLING) {
        // Step down to CRAWLING immediately, then KNEEL after 2s, then target after 2s
        planner_use_movement_mode_ = static_cast<int>(LocomotionMode::CRAWLING);
        applySpeedAndHeight(LocomotionMode::CRAWLING);
        transition_target_mode_ = LocomotionMode::IDEL_KNEEL_TWO_LEGS;
        transition_final_mode_ = target_mode;
        transition_start_time_ = std::chrono::steady_clock::now();
      } else if (current == LocomotionMode::CRAWLING) {
        // Step down to KNEEL immediately, then target after 2s
        planner_use_movement_mode_ = static_cast<int>(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
        applySpeedAndHeight(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
        transition_target_mode_ = target_mode;
        transition_start_time_ = std::chrono::steady_clock::now();
      }

      if constexpr (DEBUG_LOGGING) {
        std::cout << "[GamepadManager DEBUG] Exit crawling transition -> target: "
                  << static_cast<int>(target_mode) << std::endl;
      }
    }

    // Check if current mode is crawling or elbow crawling
    bool isInCrawlingMode() const {
      return planner_use_movement_mode_ == static_cast<int>(LocomotionMode::CRAWLING) ||
             planner_use_movement_mode_ == static_cast<int>(LocomotionMode::ELBOW_CRAWLING);
    }

    // Process timed crawling transitions
    void processCrawlingTransitions() {
      if (transition_start_time_.time_since_epoch().count() == 0) return;

      auto elapsed = std::chrono::steady_clock::now() - transition_start_time_;
      if (elapsed < std::chrono::seconds(2)) return;

      // Time to advance to the next stage
      LocomotionMode next = transition_target_mode_;
      planner_use_movement_mode_ = static_cast<int>(next);
      applySpeedAndHeight(next);

      if constexpr (DEBUG_LOGGING) {
        std::cout << "[GamepadManager DEBUG] Transition -> " << static_cast<int>(next) << std::endl;
      }

      // Check if there's a further stage (elbow crawling after crawling)
      if (transition_final_mode_ != LocomotionMode::IDLE && next != transition_final_mode_) {
        transition_target_mode_ = transition_final_mode_;
        transition_final_mode_ = LocomotionMode::IDLE;
        transition_start_time_ = std::chrono::steady_clock::now();
      } else {
        // Done
        transition_target_mode_ = LocomotionMode::IDLE;
        transition_final_mode_ = LocomotionMode::IDLE;
        transition_start_time_ = {};
      }
    }

    // Process gamepad inputs using the official motion/planner mapping.
    void processGamepadPlannerControls() {
      // Start button
      if (start_.on_press) {
        start_control_ = true;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Start pressed" << std::endl;
        }
      }

      if (left_.on_press) { delta_left_ = true; }
      if (right_.on_press) { delta_right_ = true; }

      if (!use_planner_) {
        // Motion mode: match the original Gamepad mapping.
        if (X_.on_press || Y_.on_press) {
          reinitialize_ = true;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] X/Y pressed - Reinitialize" << std::endl;
          }
        }
        if (B_.on_press) {
          motion_restart_ = true;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] B pressed - Motion restart" << std::endl;
          }
        }
        if (A_.on_press) {
          play_motion_ = true;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] A pressed - Play motion" << std::endl;
          }
        }
        if (L1_.on_press) {
          motion_prev_ = true;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] L1 pressed - Previous motion" << std::endl;
          }
        }
        if (R1_.on_press) {
          motion_next_ = true;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] R1 pressed - Next motion" << std::endl;
          }
        }
      } else {
        // Planner mode: match the original Gamepad mapping.
        if (X_.on_press || Y_.on_press) {
          reinitialize_ = true;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] X/Y pressed - Reinitialize" << std::endl;
          }
        }
        if (A_.on_press) {
          play_motion_ = true;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] A pressed - Play motion" << std::endl;
          }
        }
        if (B_.on_press) {
          planner_emergency_stop_ = true;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] B pressed - Planner emergency stop" << std::endl;
          }
        }
        if (L1_.on_press) {
          planner_use_movement_mode_ -= 1;
          if (planner_use_movement_mode_ < 0) { planner_use_movement_mode_ = 6; }
          if (planner_use_movement_mode_ == 6) { planner_use_height_ = 0.8; }
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] L1 pressed - Movement mode changed to: "
                      << planner_use_movement_mode_ << std::endl;
          }
        }
        if (R1_.on_press) {
          planner_use_movement_mode_ += 1;
          if (planner_use_movement_mode_ > 6) { planner_use_movement_mode_ = 0; }
          if (planner_use_movement_mode_ == 4) { planner_use_height_ = 0.8; }
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] R1 pressed - Movement mode changed to: "
                      << planner_use_movement_mode_ << std::endl;
          }
        }
        if (R2_.pressed) {
          if (planner_use_movement_mode_ < 4) {
            planner_use_movement_speed_ += 0.02;
            if constexpr (DEBUG_LOGGING) {
              std::cout << "[GamepadManager DEBUG] R2 pressed - Speed increasing: "
                        << planner_use_movement_speed_ << std::endl;
            }
          } else {
            planner_use_height_ += 0.02;
            if constexpr (DEBUG_LOGGING) {
              std::cout << "[GamepadManager DEBUG] R2 pressed - Height increasing: "
                        << planner_use_height_ << std::endl;
            }
          }
        }
        if (L2_.pressed) {
          if (planner_use_movement_mode_ < 4) {
            planner_use_movement_speed_ -= 0.02;
            if constexpr (DEBUG_LOGGING) {
              std::cout << "[GamepadManager DEBUG] L2 pressed - Speed decreasing: "
                        << planner_use_movement_speed_ << std::endl;
            }
          } else {
            planner_use_height_ -= 0.02;
            if constexpr (DEBUG_LOGGING) {
              std::cout << "[GamepadManager DEBUG] L2 pressed - Height decreasing: "
                        << planner_use_height_ << std::endl;
            }
          }
        }

        clampOfficialPlannerMode();

        if (std::abs(rx_) > dead_zone_ || std::abs(ry_) > dead_zone_) {
          planner_facing_angle_ = planner_facing_angle_ - 0.02 * rx_;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] Right stick - Facing angle: "
                      << planner_facing_angle_ << " rad" << std::endl;
          }
        }

        if (std::abs(lx_) > dead_zone_ || std::abs(ly_) > dead_zone_) {
          planner_moving_direction_ = atan2(ly_, lx_) - M_PI / 2 + planner_facing_angle_;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] Left stick - Moving direction: "
                      << planner_moving_direction_ << " rad" << std::endl;
          }
        }
      }
    }

    void handleGamepadMotionInput(MotionDataReader& motion_reader,
                                  std::shared_ptr<const MotionSequence>& current_motion,
                                  int& current_frame,
                                  OperatorState& operator_state,
                                  bool& reinitialize_heading,
                                  DataBuffer<HeadingState>& heading_state_buffer,
                                  std::mutex& current_motion_mutex) {
      if (motion_prev_ && !motion_reader.motions.empty()) {
        motion_reader.current_motion_index_ =
            (motion_reader.current_motion_index_ - 1 + motion_reader.motions.size()) % motion_reader.motions.size();
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          auto target_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);
          current_motion = CreateGentleReferenceMotionTransition(current_motion, current_frame, target_motion);
          current_frame = 0;
          reinitialize_heading = true;
          operator_state.play = false;
        }
      }

      if (motion_next_ && !motion_reader.motions.empty()) {
        motion_reader.current_motion_index_ = (motion_reader.current_motion_index_ + 1) % motion_reader.motions.size();
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          auto target_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);
          current_motion = CreateGentleReferenceMotionTransition(current_motion, current_frame, target_motion);
          current_frame = 0;
          reinitialize_heading = true;
          operator_state.play = false;
        }
      }

      if (play_motion_) {
        std::lock_guard<std::mutex> lock(current_motion_mutex);
        if (IsInitialStandingReferenceMotion(current_motion)) {
          auto target_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);
          current_motion = CreateGentleReferenceMotionTransition(
              current_motion, current_frame, target_motion);
          current_frame = 0;
          reinitialize_heading = true;
        }
        operator_state.play = true;
      }

      if (motion_restart_) {
        std::lock_guard<std::mutex> lock(current_motion_mutex);
        auto target_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);
        current_motion = CreateGentleReferenceMotionTransition(current_motion, current_frame, target_motion);
        current_frame = 0;
        reinitialize_heading = true;
        operator_state.play = false;
      }

      if (start_control_) { operator_state.start = true; }

      if (reinitialize_) {
        std::lock_guard<std::mutex> lock(current_motion_mutex);
        reinitialize_heading = true;
      }
    }

    void handlePlannerToggle(MotionDataReader& motion_reader,
                             std::shared_ptr<const MotionSequence>& current_motion,
                             int& current_frame,
                             OperatorState& operator_state,
                             bool& reinitialize_heading,
                             bool has_planner,
                             PlannerState& planner_state,
                             DataBuffer<MovementState>& movement_state_buffer,
                             std::mutex& current_motion_mutex) {
      if (use_planner_ && !has_planner) {
        std::cout << "[GamepadManager] Planner not loaded - cannot enable" << std::endl;
        use_planner_ = false;
      }

      if (has_planner && planner_state.enabled != use_planner_) {
        planner_state.enabled = use_planner_;
        if (planner_state.enabled) {
          std::cout << "[GamepadManager] Planner enabled" << std::endl;
          planner_facing_angle_ = 0.0;
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            operator_state.play = false;
          }

          auto wait_start = std::chrono::steady_clock::now();
          constexpr auto PLANNER_INIT_TIMEOUT = std::chrono::seconds(5);
          while (planner_state.enabled) {
            {
              std::lock_guard<std::mutex> lock(current_motion_mutex);
              if (current_motion && current_motion->name == "planner_motion") {
                std::cout << "[GamepadManager] motion name is planner_motion" << std::endl;
                break;
              }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            auto elapsed = std::chrono::steady_clock::now() - wait_start;
            if (elapsed > PLANNER_INIT_TIMEOUT) {
              std::cerr << "[GamepadManager ERROR] Planner initialization timeout after 5 seconds" << std::endl;
              use_planner_ = false;
              planner_state.enabled = false;
              movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE),
                                                          {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));
              {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                operator_state.play = false;
                if (!motion_reader.motions.empty()) {
                  current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);
                }
                current_frame = 0;
              }
              return;
            }
            std::cout << "[GamepadManager] Waiting for planner to be initialized" << std::endl;
          }

          if (!planner_state.enabled || !planner_state.initialized) {
            std::cout << "[GamepadManager] Planner failed to initialize." << std::endl;
            use_planner_ = false;
            movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE),
                                                        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));
            {
              std::lock_guard<std::mutex> lock(current_motion_mutex);
              operator_state.play = false;
              if (!motion_reader.motions.empty()) {
                current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);
              }
              current_frame = 0;
            }
          } else {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            operator_state.play = true;
          }
        } else {
          std::cout << "[GamepadManager] Planner disabled" << std::endl;
          planner_state.initialized = false;
          movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE),
                                                      {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            operator_state.play = false;
            reinitialize_heading = true;
            if (!motion_reader.motions.empty()) {
              current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);
            }
            current_frame = 0;
          }
        }
      }
    }

    void clampOfficialPlannerMode() {
      switch (planner_use_movement_mode_) {
        case 0:
          planner_use_movement_speed_ = -1.0;
          planner_use_height_ = -1.0;
          break;
        case 1:
          planner_use_movement_speed_ = std::clamp(planner_use_movement_speed_, 0.2, 0.8);
          planner_use_height_ = -1.0;
          break;
        case 2:
          planner_use_movement_speed_ = std::clamp(planner_use_movement_speed_, 0.8, 1.5);
          planner_use_height_ = -1.0;
          break;
        case 3:
          planner_use_movement_speed_ = std::clamp(planner_use_movement_speed_, 1.5, 3.0);
          planner_use_height_ = -1.0;
          break;
        case 4:
        case 5:
        case 6:
          planner_use_movement_speed_ = -1.0;
          planner_use_height_ = std::clamp(planner_use_height_, 0.1, 0.8);
          break;
      }
    }

    // Handle gamepad input (called from handle_input())
    void handleGamepadPlannerInput(MotionDataReader& motion_reader,
                                   std::shared_ptr<const MotionSequence>& current_motion,
                                   int& current_frame,
                                   OperatorState& operator_state,
                                   bool& reinitialize_heading,
                                   DataBuffer<HeadingState>& heading_state_buffer,
                                   PlannerState& planner_state,
                                   DataBuffer<MovementState>& movement_state_buffer,
                                   std::mutex& current_motion_mutex) {
      // Handle safety reset from interface manager
      if (CheckAndClearSafetyReset()) {
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          operator_state.play = false;
        }
        if (operator_state.start) {
          if (planner_state.enabled && planner_state.initialized) {
            // Planner is already on, keep it as is (don't touch initialized flag)
            {
              std::lock_guard<std::mutex> lock(current_motion_mutex);
              if (current_motion->GetEncodeMode() >= 0) {
                current_motion->SetEncodeMode(0);
              }
              operator_state.play = true;
            }
            auto current_facing = movement_state_buffer.GetDataWithTime().data->facing_direction;
            planner_facing_angle_ = std::atan2(current_facing[1], current_facing[0]);
            std::cout << "[GamepadManager] Safety reset: Planner kept enabled with current state" << std::endl;
          } else {
            // Planner was disabled, set initial movement state
            movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE), 
                                                        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));

            // Now enable planner
            planner_state.enabled = true;
            planner_facing_angle_ = 0.0;
            std::cout << "[GamepadManager] Planner enabled" << std::endl;

            // Wait for planner to be initialized with timeout (5 seconds)
            auto wait_start = std::chrono::steady_clock::now();
            constexpr auto PLANNER_INIT_TIMEOUT = std::chrono::seconds(5);
            while (planner_state.enabled) {
              {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                if (current_motion->name == "planner_motion") {
                  std::cout << "[GamepadManager] motion name is planner_motion" << std::endl;
                  break;
                }
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(100));
              auto elapsed = std::chrono::steady_clock::now() - wait_start;
              if (elapsed > PLANNER_INIT_TIMEOUT) {
                std::cerr << "[GamepadManager ERROR] Planner initialization timeout after 5 seconds" << std::endl;
                operator_state.stop = true;
                return;
              }
              std::cout << "[GamepadManager] Waiting for planner to be initialized" << std::endl;
            }

            // Check if planner is enabled and initialized
            if (!planner_state.enabled || !planner_state.initialized) {
              std::cerr << "[GamepadManager ERROR] Planner failed to initialize. Stopping control." << std::endl;
              operator_state.stop = true;
              return;
            }

            // Play motion
            {
              std::lock_guard<std::mutex> lock(current_motion_mutex);
              operator_state.play = true;
            }
          }
        }
        return;
      }

      // Handle control start (Start button)
      if (start_control_) {
        // Start control
        operator_state.start = true;
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          operator_state.play = false;
          // Reinitialize base quaternion and reset delta heading
          reinitialize_heading = true;
        }
        
        // Ensure planner is enabled (always required in GamepadManager mode)
        if (!planner_state.enabled) {
          planner_state.enabled = true;
          planner_facing_angle_ = 0.0;
          std::cout << "[GamepadManager] Planner enabled" << std::endl;
        }

        // Wait for planner to be initialized with timeout (5 seconds)
        auto wait_start = std::chrono::steady_clock::now();
        constexpr auto PLANNER_INIT_TIMEOUT = std::chrono::seconds(5);
        while (planner_state.enabled) {
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            if (current_motion->name == "planner_motion") {
              std::cout << "[GamepadManager] motion name is planner_motion" << std::endl;
              break;
            }
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          auto elapsed = std::chrono::steady_clock::now() - wait_start;
          if (elapsed > PLANNER_INIT_TIMEOUT) {
            std::cerr << "[GamepadManager ERROR] Planner initialization timeout after 5 seconds" << std::endl;
            operator_state.stop = true;
            return;
          }
          std::cout << "[GamepadManager] Waiting for planner to be initialized" << std::endl;
        }

        // Check if planner is enabled and initialized
        if (!planner_state.enabled || !planner_state.initialized) {
          std::cerr << "[GamepadManager ERROR] Planner failed to initialize. Stopping control." << std::endl;
          operator_state.stop = true;
          return;
        }

        // Play motion
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          operator_state.play = true;
        }
      }

      // Handle reinitialize command
      if (reinitialize_) {
        std::lock_guard<std::mutex> lock(current_motion_mutex);
        reinitialize_heading = true;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Reinitialized base quaternion and facing angle" << std::endl;
        }
      }

      // If planner is enabled and initialized, send movement commands
      if (planner_state.enabled && planner_state.initialized) {
        int final_mode = planner_use_movement_mode_;
        std::array<double, 3> final_movement = {double(cos(planner_moving_direction_)),
                                                 double(sin(planner_moving_direction_)), 0.0};
        std::array<double, 3> final_facing_direction = {double(cos(planner_facing_angle_)),
                                                         double(sin(planner_facing_angle_)), 0.0};
        double final_speed = planner_use_movement_speed_;
        double final_height = planner_use_height_;

        // If left sticks in dead zone — idle logic depends on motion set
        if (std::abs(lx_) < dead_zone_ && std::abs(ly_) < dead_zone_) {
          if (motion_set_index_ == 0) {
            // Standing set → go to IDLE
            final_mode = static_cast<int>(LocomotionMode::IDLE);
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = -1.0f;
            final_height = -1.0f;
          } else if (motion_set_index_ == 1) {
            // Squat set → stay in mode, zero movement
            final_movement = {0.0f, 0.0f, 0.0f};
            LocomotionMode cur = static_cast<LocomotionMode>(planner_use_movement_mode_);
            if (cur == LocomotionMode::CRAWLING || cur == LocomotionMode::ELBOW_CRAWLING) {
              final_speed = 0.0f;
            } else {
              final_speed = -1.0f;
            }
            final_height = planner_use_height_;
          } else if (motion_set_index_ == 2) {
            // Boxing set → stay in mode, zero movement
            final_movement = {0.0f, 0.0f, 0.0f};
            LocomotionMode cur = static_cast<LocomotionMode>(planner_use_movement_mode_);
            if (cur == LocomotionMode::LEFT_PUNCH || cur == LocomotionMode::RIGHT_PUNCH ||
                cur == LocomotionMode::LEFT_HOOK || cur == LocomotionMode::RIGHT_HOOK) {
              // Punches use facing direction as movement
              final_movement = final_facing_direction;
            }
            final_speed = 0.0f;
            final_height = -1.0f;
          } else if (motion_set_index_ == 3) {
            // Styled walking → stay in mode, zero movement
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = 0.0f;
            final_height = -1.0f;
          }
        }

        // Emergency stop
        if (planner_emergency_stop_) {
          if (motion_set_index_ == 0) {
            final_mode = static_cast<int>(LocomotionMode::IDLE);
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = -1.0f;
            final_height = -1.0f;
          } else {
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = 0.0f;
            final_height = (motion_set_index_ == 1) ? planner_use_height_ : -1.0f;
          }
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] Emergency stop - movement reset" << std::endl;
          }
        }

        // Static kneel modes always zero movement
        if (planner_use_movement_mode_ == static_cast<int>(LocomotionMode::IDEL_KNEEL_TWO_LEGS) ||
            planner_use_movement_mode_ == static_cast<int>(LocomotionMode::IDEL_KNEEL) ||
            planner_use_movement_mode_ == static_cast<int>(LocomotionMode::IDEL_SQUAT)) {
          final_movement = {0.0f, 0.0f, 0.0f};
          final_speed = 0.0f;
          final_height = planner_use_height_;
        }

        MovementState mode_state(final_mode, final_movement, final_facing_direction, final_speed, final_height);
        movement_state_buffer.SetData(mode_state);
      }
    }

  private:
    // ------------------------------------------------------------------
    // Owned delegate interfaces (gamepad mode is handled inline, not delegated)
    // ------------------------------------------------------------------
    std::unique_ptr<ZMQEndpointInterface> zmq_;   ///< ZMQ streaming handler.

    InputInterface* current_ = nullptr;  ///< Non-owning pointer to active delegate (nullptr in gamepad mode).

    // ------------------------------------------------------------------
    // Active-selection bookkeeping
    // ------------------------------------------------------------------
    std::vector<ManagedType> order_;         ///< Insertion-order of available modes.
    int active_index_ = 0;                   ///< Index into order_.
    ManagedType active_ = ManagedType::GAMEPAD;  ///< Currently-active mode tag.

    // ZMQ configuration (stored for deferred construction)
    std::string zmq_host_;
    int zmq_port_;
    std::string zmq_topic_;
    bool zmq_conflate_ = false;
    bool zmq_verbose_ = false;

    /// Global emergency-stop flag, set by 'O'/'o' keyboard shortcut.
    bool emergency_stop_ = false;
    /// Global temperature report flag, set by 'F'/'f' keyboard shortcut.
    bool report_temperature_flag_ = false;

    // ------------------------------------------------------------------
    // Per-frame gamepad action flags (reset at the start of update())
    // ------------------------------------------------------------------
    bool start_control_ = false;           ///< Start-button pressed this frame.
    bool stop_control_ = false;            ///< Select-button pressed this frame.
    bool motion_prev_ = false;             ///< Switch to previous reference motion.
    bool motion_next_ = false;             ///< Switch to next reference motion.
    bool play_motion_ = false;             ///< Start / resume reference motion playback.
    bool motion_restart_ = false;          ///< Reset current reference motion to frame 0.
    bool delta_left_ = false;              ///< Nudge heading left in motion mode.
    bool delta_right_ = false;             ///< Nudge heading right in motion mode.
    bool reinitialize_ = false;            ///< X/Y-button reinitialize heading.
    bool planner_emergency_stop_ = false;  ///< A-button emergency stop.

    // ------------------------------------------------------------------
    // Gamepad hardware state
    // ------------------------------------------------------------------
    unitree::common::REMOTE_DATA_RX gamepad_data_ = unitree::common::REMOTE_DATA_RX();  ///< Raw 40-byte packet.

    // Smoothed analog stick values
    float lx_ = 0.0f;   ///< Left stick horizontal (smoothed).
    float rx_ = 0.0f;   ///< Right stick horizontal (smoothed).
    float ry_ = 0.0f;   ///< Right stick vertical (smoothed).
    float l2_ = 0.0f;   ///< Left trigger analog (smoothed).
    float ly_ = 0.0f;   ///< Left stick vertical (smoothed).
    float smooth_ = 0.3f;       ///< EMA smoothing factor.
    float dead_zone_ = 0.05f;   ///< Analog dead-zone threshold.

    // Edge-detecting buttons
    unitree::common::Button R1_, L1_, start_, select_, R2_, L2_;
    unitree::common::Button F1_, F2_, A_, B_, X_, Y_;
    unitree::common::Button up_, right_, down_, left_;

    // ------------------------------------------------------------------
    // Motion set state
    // ------------------------------------------------------------------
    int motion_set_index_ = 0;                    ///< Active motion set (0=Standing, 1=Squat, 2=Boxing, 3=Styled).
    std::vector<LocomotionMode> current_motion_set_;  ///< Modes in the active set.
    int mode_index_in_set_ = 0;                   ///< Current mode index within the set.

    // ------------------------------------------------------------------
    // Planner control state (persists across frames)
    // ------------------------------------------------------------------
    bool use_planner_ = false;  ///< True while F1-selected planner mode is active.
    int planner_use_movement_mode_ = static_cast<int>(LocomotionMode::SLOW_WALK);  ///< Current locomotion mode.
    double planner_use_movement_speed_ = -1.0;   ///< Desired speed (-1 = mode default).
    double planner_use_height_ = -1.0;            ///< Desired body height (−1 = mode default).
    double planner_facing_angle_ = 0.0;           ///< Accumulated facing direction (radians).
    double planner_moving_direction_ = 0.0;       ///< Current movement direction (radians).

    // ------------------------------------------------------------------
    // Staged crawling transition state
    // ------------------------------------------------------------------
    /// Target mode for the current transition stage (e.g., CRAWLING after kneeling).
    LocomotionMode transition_target_mode_ = LocomotionMode::IDLE;
    /// Final mode for multi-stage transitions (e.g., ELBOW_CRAWLING via kneel→crawl→elbow).
    LocomotionMode transition_final_mode_ = LocomotionMode::IDLE;
    /// Timestamp when the current transition stage began.
    std::chrono::time_point<std::chrono::steady_clock> transition_start_time_{};

    // ------------------------------------------------------------------
    // Boxing auto-revert state
    // ------------------------------------------------------------------
    /// Timestamp when a punch/hook/idle was triggered; reverts to WALK_BOXING after 1s.
    std::chrono::time_point<std::chrono::steady_clock> boxing_revert_time_{};
};

#endif // GAMEPAD_MANAGER_HPP
