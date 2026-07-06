#include "turbo_serial.h"

#include <stdio.h>

/* Example of how to get a snapshot of serial ports through the TurboSerial ABI. */

int main(void) {
  turbo_serial_port_list_t *ports = NULL;
  turbo_serial_result_t result;
  size_t count;
  size_t i;

  result = turbo_serial_list_ports(&ports);
  if (result != TURBO_SERIAL_OK) {
    printf("turbo_serial_list_ports() failed: %s\n", turbo_serial_result_name(result));
    return 1;
  }

  count = turbo_serial_port_list_count(ports);
  for (i = 0; i < count; ++i) {
    const turbo_serial_port_info_t *info = turbo_serial_port_list_get(ports, i);
    if (info && info->name) {
      printf("Found port: %s\n", info->name);
    }
  }

  printf("Found %zu ports.\n", count);
  turbo_serial_port_list_destroy(ports);
  return 0;
}
