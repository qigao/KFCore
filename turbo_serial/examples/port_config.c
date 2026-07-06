#include "turbo_serial.h"

#include <stdio.h>

/* Example of configuring a serial handle through the TurboSerial ABI. */

int main(int argc, char **argv) {
  turbo_serial_config_t config;
  turbo_serial_t *serial = NULL;
  turbo_serial_result_t result;

  if (argc != 2) {
    printf("Usage: %s <port name>\n", argv[0]);
    return 1;
  }

  turbo_serial_config_default(&config);
  config.baudrate = 115200;
  config.bits = 8;
  config.parity = TURBO_SERIAL_PARITY_NONE;
  config.stopbits = 1;
  config.flowcontrol = TURBO_SERIAL_FLOWCONTROL_NONE;
  config.rx_buffer_size = 4096;
  config.tx_buffer_size = 4096;

  result = turbo_serial_create(&serial, &config);
  if (result != TURBO_SERIAL_OK) {
    printf("turbo_serial_create() failed: %s\n", turbo_serial_result_name(result));
    return 1;
  }

  result = turbo_serial_open(serial, argv[1], TURBO_SERIAL_MODE_READ_WRITE);
  if (result != TURBO_SERIAL_OK) {
    printf("turbo_serial_open() failed: %s\n", turbo_serial_result_name(result));
    turbo_serial_destroy(serial);
    return 1;
  }

  printf("Opened %s at %d baud, %d data bits, %d stop bit.\n", argv[1], config.baudrate,
         config.bits, config.stopbits);

  turbo_serial_destroy(serial);
  return 0;
}
