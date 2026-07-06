#include "turbo_serial.h"
#include "tinytest.h"

suite("turbo_serial public ABI") {
  it("uses only turbo serial public types") {
    turbo_serial_config_t config;
    turbo_serial_t *serial = NULL;

    turbo_serial_config_default(&config);

    check_int_eq(config.parity, TURBO_SERIAL_PARITY_NONE);
    check_int_eq(config.flowcontrol, TURBO_SERIAL_FLOWCONTROL_NONE);
    check_int_eq(turbo_serial_create(&serial, &config), TURBO_SERIAL_OK);
    check_not_null(serial);
    check_int_eq(turbo_serial_open(serial, "not-a-real-port", (turbo_serial_mode_t)0),
                 TURBO_SERIAL_INVALID_VALUE);

    turbo_serial_destroy(serial);
  }

  it("exposes port metadata without libserialport types") {
    turbo_serial_port_list_t *ports = NULL;
    const turbo_serial_port_info_t *info = NULL;
    turbo_serial_result_t result;
    size_t count;

    result = turbo_serial_list_ports(&ports);
    check(result == TURBO_SERIAL_OK || result == TURBO_SERIAL_NOT_SUPPORTED);

    if (result == TURBO_SERIAL_OK) {
      count = turbo_serial_port_list_count(ports);
      if (count > 0) {
        info = turbo_serial_port_list_get(ports, 0);
        check_not_null(info);
        check_not_null(info->name);
        check(info->transport == TURBO_SERIAL_TRANSPORT_NATIVE ||
              info->transport == TURBO_SERIAL_TRANSPORT_USB ||
              info->transport == TURBO_SERIAL_TRANSPORT_BLUETOOTH ||
              info->transport == TURBO_SERIAL_TRANSPORT_UNKNOWN);
      }
      check_null(turbo_serial_port_list_get(ports, count));
      turbo_serial_port_list_destroy(ports);
    } else {
      check_null(ports);
    }

    check_int_eq(turbo_serial_port_info_by_name(NULL, &ports, &info),
                 TURBO_SERIAL_INVALID_VALUE);
    check_null(ports);
    check_null(info);
  }

  it("exposes event sets without libserialport types") {
    turbo_serial_event_set_t *event_set = NULL;
    turbo_serial_t *serial = NULL;

    check_int_eq(turbo_serial_event_set_create(&event_set), TURBO_SERIAL_OK);
    check_not_null(event_set);

    check_int_eq(turbo_serial_create(&serial, NULL), TURBO_SERIAL_OK);
    check_not_null(serial);
    check_int_eq(turbo_serial_event_set_add(event_set, serial, TURBO_SERIAL_EVENT_RX_READY),
                 TURBO_SERIAL_INVALID_STATE);
    check_int_eq(turbo_serial_event_set_add(event_set, serial, 0x8000u),
                 TURBO_SERIAL_INVALID_VALUE);
    check_int_eq(turbo_serial_event_wait(event_set, 1), TURBO_SERIAL_INVALID_STATE);

    turbo_serial_destroy(serial);
    turbo_serial_event_set_destroy(event_set);
  }
}
