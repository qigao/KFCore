#include "serialport.h"
#include "tinytest.h"

#include <string.h>

suite("libserialport") {
  it("reports vendored version") {
    check_int_eq(sp_get_major_package_version(), SP_PACKAGE_VERSION_MAJOR);
    check_int_eq(sp_get_minor_package_version(), SP_PACKAGE_VERSION_MINOR);
    check_int_eq(sp_get_micro_package_version(), SP_PACKAGE_VERSION_MICRO);
    check_str_eq(sp_get_package_version_string(), SP_PACKAGE_VERSION_STRING);
  }

  it("allocates and updates a port configuration") {
    struct sp_port_config *config = NULL;
    int baudrate = 0;
    int bits = 0;
    enum sp_parity parity = SP_PARITY_INVALID;

    check_int_eq(sp_new_config(&config), SP_OK);
    check_not_null(config);

    check_int_eq(sp_set_config_baudrate(config, 115200), SP_OK);
    check_int_eq(sp_get_config_baudrate(config, &baudrate), SP_OK);
    check_int_eq(baudrate, 115200);

    check_int_eq(sp_set_config_bits(config, 8), SP_OK);
    check_int_eq(sp_get_config_bits(config, &bits), SP_OK);
    check_int_eq(bits, 8);

    check_int_eq(sp_set_config_parity(config, SP_PARITY_NONE), SP_OK);
    check_int_eq(sp_get_config_parity(config, &parity), SP_OK);
    check_int_eq(parity, SP_PARITY_NONE);

    check_int_eq(sp_set_config_flowcontrol(config, SP_FLOWCONTROL_NONE), SP_OK);
    sp_free_config(config);
  }

  it("validates null arguments") {
    check_int_eq(sp_new_config(NULL), SP_ERR_ARG);
    check_null(sp_get_port_name(NULL));
    check_int_eq(sp_get_port_by_name(NULL, NULL), SP_ERR_ARG);
    check_int_eq(sp_list_ports(NULL), SP_ERR_ARG);
  }

  it("enumerates ports or reports unsupported enumeration") {
    struct sp_port **ports = NULL;
    enum sp_return result = sp_list_ports(&ports);

    check(result == SP_OK || result == SP_ERR_SUPP);

    if (result == SP_OK) {
      check_not_null(ports);
      sp_free_port_list(ports);
    } else {
      check_null(ports);
    }
  }

  it("keeps libserialport timeout semantics on TurboNet monotonic time") {
    struct timeout timeout;

    timeout_start(&timeout, 1);
    check(!timeout_check(&timeout));

    timeout_update(&timeout);
    check(!timeout_check(&timeout) || timeout_remaining_ms(&timeout) <= 1);

    timeout_start(&timeout, 0);
    timeout_update(&timeout);
    check(!timeout_check(&timeout));
  }

  it("keeps turbo serial public configuration independent from libserialport ABI") {
    turbo_serial_config_t config;

    turbo_serial_config_default(&config);

    check_int_eq(config.parity, TURBO_SERIAL_PARITY_NONE);
    check_int_eq(config.flowcontrol, TURBO_SERIAL_FLOWCONTROL_NONE);
    check_int_eq(TURBO_SERIAL_MODE_READ, 1);
    check_int_eq(TURBO_SERIAL_MODE_WRITE, 2);
    check_int_eq(TURBO_SERIAL_MODE_READ_WRITE, 3);
    check_str_eq(turbo_serial_result_name(TURBO_SERIAL_WOULD_BLOCK), "would block");
  }

  it("creates a per-instance turbo serial buffer") {
    turbo_serial_config_t config;
    turbo_serial_t *serial = NULL;

    turbo_serial_config_default(&config);
    check_size_eq(config.io_chunk_size, 256);
    check_int_eq((int)config.poll_interval_ms, 10);
    config.rx_buffer_size = 16;
    config.tx_buffer_size = 16;

    check_int_eq(turbo_serial_create(&serial, &config), TURBO_SERIAL_OK);
    check_not_null(serial);
    check_size_eq(turbo_serial_rx_capacity(serial), 15);
    check_size_eq(turbo_serial_tx_capacity(serial), 15);
    check_size_eq(turbo_serial_rx_available(serial), 0);
    check_size_eq(turbo_serial_tx_available(serial), 0);

    turbo_serial_destroy(serial);
  }

  it("rejects invalid turbo serial buffer sizes") {
    turbo_serial_config_t config;
    turbo_serial_t *serial = NULL;

    turbo_serial_config_default(&config);
    config.rx_buffer_size = 15;

    check_int_eq(turbo_serial_create(&serial, &config), TURBO_SERIAL_INVALID_VALUE);
    check_null(serial);

    turbo_serial_config_default(&config);
    config.io_chunk_size = 0;
    check_int_eq(turbo_serial_create(&serial, &config), TURBO_SERIAL_INVALID_VALUE);
    check_null(serial);

    turbo_serial_config_default(&config);
    config.poll_interval_ms = 0;
    check_int_eq(turbo_serial_create(&serial, &config), TURBO_SERIAL_INVALID_VALUE);
    check_null(serial);
  }

  it("requires an open port before starting async pump") {
    turbo_serial_t *serial = NULL;

    check_int_eq(turbo_serial_create(&serial, NULL), TURBO_SERIAL_OK);
    check_not_null(serial);

    check_int_eq(turbo_serial_start_async(serial), TURBO_SERIAL_INVALID_STATE);
    check_int_eq(turbo_serial_async_running(serial), 0);
    check_int_eq(turbo_serial_stop_async(serial), TURBO_SERIAL_OK);
    check_int_eq(turbo_serial_last_error(serial), TURBO_SERIAL_OK);

    turbo_serial_destroy(serial);
  }

  it("rejects invalid turbo serial open mode before touching the port") {
    turbo_serial_t *serial = NULL;

    check_int_eq(turbo_serial_create(&serial, NULL), TURBO_SERIAL_OK);
    check_not_null(serial);

    check_int_eq(turbo_serial_open(serial, "not-a-real-port", (turbo_serial_mode_t)99),
                 TURBO_SERIAL_INVALID_VALUE);

    turbo_serial_destroy(serial);
  }

  it("buffers rx and tx data through per-handle SPSC rings") {
    turbo_serial_config_t config;
    turbo_serial_t *serial = NULL;
    const char rx_data[] = "sensor:1234";
    const char tx_data[] = "command:go";
    char out[32] = {0};
    size_t count = 0;

    turbo_serial_config_default(&config);
    config.rx_buffer_size = 32;
    config.tx_buffer_size = 32;

    check_int_eq(turbo_serial_create(&serial, &config), TURBO_SERIAL_OK);
    check_not_null(serial);

    check_int_eq(turbo_serial_buffer_rx(serial, rx_data, strlen(rx_data), &count),
                 TURBO_SERIAL_OK);
    check_size_eq(count, strlen(rx_data));
    check_size_eq(turbo_serial_rx_available(serial), strlen(rx_data));

    check_int_eq(turbo_serial_read_buffered(serial, out, sizeof(out), &count),
                 TURBO_SERIAL_OK);
    check_size_eq(count, strlen(rx_data));
    check_mem_eq(out, rx_data, strlen(rx_data));
    check_size_eq(turbo_serial_rx_available(serial), 0);

    memset(out, 0, sizeof(out));
    check_int_eq(turbo_serial_write_buffered(serial, tx_data, strlen(tx_data), &count),
                 TURBO_SERIAL_OK);
    check_size_eq(count, strlen(tx_data));
    check_size_eq(turbo_serial_tx_available(serial), strlen(tx_data));

    check_int_eq(turbo_serial_drain_tx_buffer(serial, out, sizeof(out), &count),
                 TURBO_SERIAL_OK);
    check_size_eq(count, strlen(tx_data));
    check_mem_eq(out, tx_data, strlen(tx_data));
    check_size_eq(turbo_serial_tx_available(serial), 0);

    turbo_serial_destroy(serial);
  }

  it("reports would-block when buffered reads or writes cannot progress") {
    turbo_serial_config_t config;
    turbo_serial_t *serial = NULL;
    char out[4] = {0};
    size_t count = 99;

    turbo_serial_config_default(&config);
    config.rx_buffer_size = 8;
    config.tx_buffer_size = 8;

    check_int_eq(turbo_serial_create(&serial, &config), TURBO_SERIAL_OK);
    check_not_null(serial);

    check_int_eq(turbo_serial_read_buffered(serial, out, sizeof(out), &count),
                 TURBO_SERIAL_WOULD_BLOCK);
    check_size_eq(count, 0);

    check_int_eq(turbo_serial_write_buffered(serial, "1234567", 7, &count), TURBO_SERIAL_OK);
    check_size_eq(count, 7);
    check_int_eq(turbo_serial_write_buffered(serial, "x", 1, &count),
                 TURBO_SERIAL_WOULD_BLOCK);
    check_size_eq(count, 0);

    turbo_serial_destroy(serial);
  }

  it("clears byte counts on buffered argument errors") {
    turbo_serial_t *serial = NULL;
    size_t count = 99;

    check_int_eq(turbo_serial_create(&serial, NULL), TURBO_SERIAL_OK);
    check_not_null(serial);

    check_int_eq(turbo_serial_write_buffered(serial, NULL, 1, &count),
                 TURBO_SERIAL_INVALID_VALUE);
    check_size_eq(count, 0);

    count = 99;
    check_int_eq(turbo_serial_read_buffered(serial, NULL, 1, &count),
                 TURBO_SERIAL_INVALID_VALUE);
    check_size_eq(count, 0);

    turbo_serial_destroy(serial);
  }
}
