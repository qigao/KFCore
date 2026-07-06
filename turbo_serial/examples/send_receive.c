#include "turbo_serial.h"

#include <stdio.h>
#include <string.h>

/* Example of blocking write/read through one open TurboSerial handle. */

int main(int argc, char **argv) {
  turbo_serial_config_t config;
  turbo_serial_t *serial = NULL;
  const char data[] = "Hello!";
  char buffer[sizeof(data)] = {0};
  size_t bytes = 0;
  turbo_serial_result_t result;

  if (argc != 2) {
    printf("Usage: %s <port name>\n", argv[0]);
    return 1;
  }

  turbo_serial_config_default(&config);
  config.baudrate = 9600;

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

  result = turbo_serial_write(serial, data, strlen(data), 1000, &bytes);
  if (result != TURBO_SERIAL_OK) {
    printf("turbo_serial_write() failed: %s\n", turbo_serial_result_name(result));
    turbo_serial_destroy(serial);
    return 1;
  }
  printf("Wrote %zu bytes.\n", bytes);

  result = turbo_serial_read(serial, buffer, strlen(data), 1000, &bytes);
  if (result != TURBO_SERIAL_OK) {
    printf("turbo_serial_read() failed: %s\n", turbo_serial_result_name(result));
    turbo_serial_destroy(serial);
    return 1;
  }

  printf("Read %zu bytes: %.*s\n", bytes, (int)bytes, buffer);
  turbo_serial_destroy(serial);
  return 0;
}
