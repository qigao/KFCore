set(TURBO_SERIAL_EXAMPLES
    await_events
    handle_errors
    list_ports
    port_config
    port_info
    send_receive)

add_custom_target(${EXAMPLES_TARGET_NAME})
set_target_properties(${EXAMPLES_TARGET_NAME} PROPERTIES FOLDER "turbo_serial/examples")

foreach(example IN LISTS TURBO_SERIAL_EXAMPLES)
  add_executable(example_${example} EXCLUDE_FROM_ALL ${example}.c)
  target_link_libraries(example_${example} PRIVATE ${TARGET_NAME})
  add_dependencies(${EXAMPLES_TARGET_NAME} example_${example})
  set_target_properties(example_${example} PROPERTIES FOLDER "turbo_serial/examples")
endforeach()
