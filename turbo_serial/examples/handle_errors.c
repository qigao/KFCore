#include "turbo_serial.h"

#include <stdio.h>

/* Example of handling TurboSerial result codes at the application boundary. */

static int check(turbo_serial_result_t result, const char *operation) {
  if (result == TURBO_SERIAL_OK) return 0;

  printf("%s failed: %s\n", operation, turbo_serial_result_name(result));
  return 1;
}

int main(void) {
  turbo_serial_port_list_t *ports = NULL;
  turbo_serial_t *serial = NULL;
  turbo_serial_result_t result;

  result = turbo_serial_list_ports(&ports);
  if (check(result, "turbo_serial_list_ports")) return 1;
  turbo_serial_port_list_destroy(ports);

  result = turbo_serial_create(&serial, NULL);
  if (check(result, "turbo_serial_create")) return 1;

  result = turbo_serial_open(serial, "NON-EXISTENT-PORT", TURBO_SERIAL_MODE_READ_WRITE);
  if (result != TURBO_SERIAL_OK) {
    printf("Expected open failure: %s\n", turbo_serial_result_name(result));
  }

  turbo_serial_destroy(serial);
  return 0;
}
