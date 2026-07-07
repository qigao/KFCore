#ifndef TURBO_SERIAL_H
#define TURBO_SERIAL_H

#include <stddef.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(TURBO_SERIAL_EXPORTS)
    #define TURBO_SERIAL_API __declspec(dllexport)
  #else
    #define TURBO_SERIAL_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__) && __GNUC__ >= 4
  #define TURBO_SERIAL_API __attribute__((visibility("default")))
#else
  #define TURBO_SERIAL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque serial handle owned by turbo_serial_create()/turbo_serial_destroy(). */
typedef struct turbo_serial_t turbo_serial_t;

/* Opaque port-list snapshot owned by turbo_serial_port_list_destroy(). */
typedef struct turbo_serial_port_list turbo_serial_port_list_t;

/* Opaque event set owned by turbo_serial_event_set_destroy(). */
typedef struct turbo_serial_event_set turbo_serial_event_set_t;

/* Public result codes returned by the TurboSerial ABI. */
typedef enum turbo_serial_result {
  TURBO_SERIAL_OK = 0,
  TURBO_SERIAL_INVALID_VALUE,
  TURBO_SERIAL_INVALID_STATE,
  TURBO_SERIAL_IO_FAILED,
  TURBO_SERIAL_NO_MEMORY,
  TURBO_SERIAL_NOT_SUPPORTED,
  TURBO_SERIAL_WOULD_BLOCK
} turbo_serial_result_t;

/* Access requested when opening a serial port. */
typedef enum turbo_serial_mode {
  TURBO_SERIAL_MODE_READ = 1,
  TURBO_SERIAL_MODE_WRITE = 2,
  TURBO_SERIAL_MODE_READ_WRITE = 3
} turbo_serial_mode_t;

/* Parity policy used when configuring an opened port. */
typedef enum turbo_serial_parity {
  TURBO_SERIAL_PARITY_INVALID = -1,
  TURBO_SERIAL_PARITY_NONE = 0,
  TURBO_SERIAL_PARITY_ODD = 1,
  TURBO_SERIAL_PARITY_EVEN = 2,
  TURBO_SERIAL_PARITY_MARK = 3,
  TURBO_SERIAL_PARITY_SPACE = 4
} turbo_serial_parity_t;

/* Common flow-control presets. */
typedef enum turbo_serial_flowcontrol {
  TURBO_SERIAL_FLOWCONTROL_NONE = 0,
  TURBO_SERIAL_FLOWCONTROL_XONXOFF = 1,
  TURBO_SERIAL_FLOWCONTROL_RTSCTS = 2,
  TURBO_SERIAL_FLOWCONTROL_DTRDSR = 3
} turbo_serial_flowcontrol_t;

/* Transport class reported by port metadata. */
typedef enum turbo_serial_transport {
  TURBO_SERIAL_TRANSPORT_UNKNOWN = 0,
  TURBO_SERIAL_TRANSPORT_NATIVE,
  TURBO_SERIAL_TRANSPORT_USB,
  TURBO_SERIAL_TRANSPORT_BLUETOOTH
} turbo_serial_transport_t;

/* Event bits accepted by turbo_serial_event_set_add(). */
typedef enum turbo_serial_event {
  TURBO_SERIAL_EVENT_RX_READY = 1,
  TURBO_SERIAL_EVENT_TX_READY = 2,
  TURBO_SERIAL_EVENT_ERROR = 4
} turbo_serial_event_t;

/* Read-only port metadata. String pointers remain valid until the owning
 * turbo_serial_port_list_t is destroyed.
 */
typedef struct turbo_serial_port_info {
  const char *name;
  const char *description;
  turbo_serial_transport_t transport;

  int has_usb_bus_address;
  int usb_bus;
  int usb_address;

  int has_usb_vid_pid;
  int usb_vid;
  int usb_pid;

  const char *usb_manufacturer;
  const char *usb_product;
  const char *usb_serial;
  const char *bluetooth_address;
} turbo_serial_port_info_t;

/* Configuration copied into each handle at creation time.
 *
 * rx_buffer_size and tx_buffer_size must be powers of two and at least 2.
 * io_chunk_size and poll_interval_ms must be non-zero.
 */
typedef struct turbo_serial_config {
  int baudrate;
  int bits;
  turbo_serial_parity_t parity;
  int stopbits;
  turbo_serial_flowcontrol_t flowcontrol;
  size_t rx_buffer_size;
  size_t tx_buffer_size;
  size_t io_chunk_size;
  unsigned int poll_interval_ms;
} turbo_serial_config_t;

/* Fill config with portable defaults: 115200 8N1, no flow control. */
TURBO_SERIAL_API void turbo_serial_config_default(turbo_serial_config_t *config);

/* Return a stable static name for a TurboSerial result code. */
TURBO_SERIAL_API const char *turbo_serial_result_name(turbo_serial_result_t result);

/* Create a snapshot of ports currently visible to the system. */
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_list_ports(turbo_serial_port_list_t **ports);

/* Create a one-port metadata snapshot by system port name. */
TURBO_SERIAL_API turbo_serial_result_t
turbo_serial_port_info_by_name(const char *port_name, turbo_serial_port_list_t **ports,
                               const turbo_serial_port_info_t **info);

/* Release a port-list snapshot and all string views returned from it. */
TURBO_SERIAL_API void turbo_serial_port_list_destroy(turbo_serial_port_list_t *ports);

/* Query a port-list snapshot. A NULL result from get means index is out of range. */
TURBO_SERIAL_API size_t turbo_serial_port_list_count(const turbo_serial_port_list_t *ports);
TURBO_SERIAL_API const turbo_serial_port_info_t *
turbo_serial_port_list_get(const turbo_serial_port_list_t *ports, size_t index);

/* Create a closed handle. The supplied config is copied and may be NULL for defaults. */
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_create(turbo_serial_t **serial,
                                                          const turbo_serial_config_t *config);

/* Stop async I/O if needed, close the port, and release the handle. */
TURBO_SERIAL_API void turbo_serial_destroy(turbo_serial_t *serial);

/* Open a named serial port and apply the handle configuration. */
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_open(turbo_serial_t *serial,
                                                        const char *port_name,
                                                        turbo_serial_mode_t mode);

/* Stop async I/O if needed and close the current port. */
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_close(turbo_serial_t *serial);

/* Blocking read/write calls use the OS serial handle directly. They are valid
 * only while the async pump is stopped.
 */
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_read(turbo_serial_t *serial, void *buf,
                                                        size_t count, unsigned int timeout_ms,
                                                        size_t *bytes_read);
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_write(turbo_serial_t *serial,
                                                         const void *buf, size_t count,
                                                         unsigned int timeout_ms,
                                                         size_t *bytes_written);

/* Start or stop the background pump that moves bytes between the port and SPSC buffers. */
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_start_async(turbo_serial_t *serial);
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_stop_async(turbo_serial_t *serial);

/* Query async pump state and the last non-OK async pump error. */
TURBO_SERIAL_API int turbo_serial_async_running(const turbo_serial_t *serial);
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_last_error(const turbo_serial_t *serial);

/* Event sets wait on currently opened TurboSerial handles.
 * A timeout is reported as TURBO_SERIAL_WOULD_BLOCK.
 */
TURBO_SERIAL_API turbo_serial_result_t
turbo_serial_event_set_create(turbo_serial_event_set_t **event_set);
TURBO_SERIAL_API void turbo_serial_event_set_destroy(turbo_serial_event_set_t *event_set);
TURBO_SERIAL_API turbo_serial_result_t
turbo_serial_event_set_add(turbo_serial_event_set_t *event_set, const turbo_serial_t *serial,
                           unsigned int events);
TURBO_SERIAL_API turbo_serial_result_t
turbo_serial_event_wait(turbo_serial_event_set_t *event_set, unsigned int timeout_ms);

/* Buffered byte counts and capacities exclude the reserved SPSC sentinel slot. */
TURBO_SERIAL_API size_t turbo_serial_rx_available(const turbo_serial_t *serial);
TURBO_SERIAL_API size_t turbo_serial_tx_available(const turbo_serial_t *serial);
TURBO_SERIAL_API size_t turbo_serial_rx_capacity(const turbo_serial_t *serial);
TURBO_SERIAL_API size_t turbo_serial_tx_capacity(const turbo_serial_t *serial);

/* Buffered APIs operate on the handle-owned SPSC queues and are the async data
 * path used with turbo_serial_start_async().
 *
 * Threading contract per handle:
 * - RX ring: one producer is the async pump, one consumer calls read_buffered.
 * - TX ring: one producer calls write_buffered, one consumer is the async pump.
 * Multiple application readers or multiple application writers require
 * external synchronization.
 *
 * turbo_serial_buffer_rx() and turbo_serial_drain_tx_buffer() are intended for
 * tests and adapters and require the async pump to be stopped.
 */
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_buffer_rx(turbo_serial_t *serial,
                                                             const void *buf, size_t count,
                                                             size_t *bytes_buffered);
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_read_buffered(turbo_serial_t *serial,
                                                                 void *buf, size_t count,
                                                                 size_t *bytes_read);
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_write_buffered(turbo_serial_t *serial,
                                                                  const void *buf, size_t count,
                                                                  size_t *bytes_buffered);
TURBO_SERIAL_API turbo_serial_result_t turbo_serial_drain_tx_buffer(turbo_serial_t *serial,
                                                                   void *buf, size_t count,
                                                                   size_t *bytes_read);

#ifdef __cplusplus
}
#endif


#endif /* TURBO_SERIAL_H */
