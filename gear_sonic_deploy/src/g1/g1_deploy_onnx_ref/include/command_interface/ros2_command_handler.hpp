/**
 * @file ros2_command_handler.hpp
 * @brief Global ROS 2 command receiver for reference-motion selection and stop.
 *
 * This handler is intentionally separate from InputInterface.  Selecting a
 * dance and stopping the robot are global commands and must not depend on the
 * currently active input source.
 */

#ifndef ROS2_COMMAND_HANDLER_HPP
#define ROS2_COMMAND_HANDLER_HPP

#if HAS_ROS2

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

class ROS2CommandHandler {
 public:
  explicit ROS2CommandHandler(const std::string& node_name = "g1_command_handler") {
    if (!rclcpp::ok()) { rclcpp::init(0, nullptr); }
    this->node_ = rclcpp::Node::make_shared(node_name);
    this->SetupInterfaces();
    std::cout << "[ROS2CommandHandler] Node initialized: " << node_name << std::endl;
  }

  void Update() {
    if (this->node_ && rclcpp::ok()) { rclcpp::spin_some(this->node_); }
  }

  void PublishMotionCatalog(const std::string& catalog_json) {
    if (!this->motion_catalog_pub_) { return; }
    std_msgs::msg::String msg;
    msg.data = catalog_json;
    this->motion_catalog_pub_->publish(msg);
    std::cout << "[ROS2CommandHandler] Published motion catalog" << std::endl;
  }

  bool ConsumeSelectMotion(std::string& request) {
    std::lock_guard<std::mutex> lock(this->command_mutex_);
    if (!this->pending_select_motion_) { return false; }
    request = this->pending_motion_request_;
    this->pending_motion_request_.clear();
    this->pending_select_motion_ = false;
    return true;
  }

  bool ConsumeEmergencyStop() {
    return this->pending_emergency_stop_.exchange(false);
  }

 private:
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr select_motion_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr emergency_stop_srv_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr motion_catalog_pub_;

  std::mutex command_mutex_;
  bool pending_select_motion_ = false;
  std::string pending_motion_request_;
  std::atomic<bool> pending_emergency_stop_{false};

  void SetupInterfaces() {
    this->select_motion_sub_ = this->node_->create_subscription<std_msgs::msg::String>(
      "/WBCPolicy/select_motion",
      10,
      [this](std_msgs::msg::String::SharedPtr msg) {
        this->HandleSelectMotion(msg);
      }
    );

    this->emergency_stop_srv_ = this->node_->create_service<std_srvs::srv::Trigger>(
      "/WBCPolicy/emergency_stop",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        (void)request;
        this->pending_emergency_stop_.store(true);
        response->success = true;
        response->message = "Emergency stop queued";
        std::cout << "[ROS2CommandHandler] Emergency stop requested" << std::endl;
      }
    );

    rclcpp::QoS catalog_qos(1);
    catalog_qos.transient_local();
    catalog_qos.reliable();
    this->motion_catalog_pub_ = this->node_->create_publisher<std_msgs::msg::String>(
      "/WBCPolicy/motion_catalog",
      catalog_qos
    );
  }

  void HandleSelectMotion(const std_msgs::msg::String::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(this->command_mutex_);
    this->pending_motion_request_ = msg->data;
    this->pending_select_motion_ = true;
    std::cout << "[ROS2CommandHandler] Select motion requested: " << msg->data << std::endl;
  }
};

#endif // HAS_ROS2

#endif // ROS2_COMMAND_HANDLER_HPP
