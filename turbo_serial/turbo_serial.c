#define TURBOSERIAL_API_ONLY
#include "serialport.h"

#include "ring_buffer_spsc.h"
#include "platform.h"
#include "turbo_thread.h"

#include <stdbool.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct turbo_serial_base {
  turbo_serial_config_t config;
} turbo_serial_base_t;

typedef struct turbo_serial_port {
  struct sp_port *port;
} turbo_serial_port_t;

typedef struct turbo_serial_rings {
  ring_spsc_t rx_ring;
  ring_spsc_t tx_ring;
  uint8_t *rx_storage;
  uint8_t *tx_storage;
} turbo_serial_rings_t;

typedef struct turbo_serial_async {
  turbo_thread_t worker_thread;
  turbo_timer_t *pump_timer;
  turbo_mutex_t worker_mutex;
  turbo_cond_t worker_cond;
  atomic_bool async_running;
  atomic_bool wake_requested;
} turbo_serial_async_t;

typedef struct turbo_serial_error {
  atomic_int last_error;
} turbo_serial_error_t;

typedef struct turbo_serial_port_info_storage {
  turbo_serial_port_info_t view;
  char *name;
  char *description;
  char *usb_manufacturer;
  char *usb_product;
  char *usb_serial;
  char *bluetooth_address;
} turbo_serial_port_info_storage_t;

typedef struct turbo_serial_backend_ops turbo_serial_backend_ops_t;

struct turbo_serial_port_list {
  turbo_serial_port_info_storage_t *items;
  size_t count;
};

struct turbo_serial_event_set {
  const turbo_serial_backend_ops_t *ops;
  struct sp_event_set *set;
  size_t count;
};

typedef struct turbo_serial_size_result {
  turbo_serial_result_t code;
  size_t count;
} turbo_serial_size_result_t;

struct turbo_serial_backend_ops {
  enum sp_return (*list_ports)(struct sp_port ***list_ptr);
  enum sp_return (*get_port_by_name)(const char *portname, struct sp_port **port_ptr);
  void (*free_port)(struct sp_port *port);
  void (*free_port_list)(struct sp_port **ports);

  char *(*get_port_name)(const struct sp_port *port);
  char *(*get_port_description)(const struct sp_port *port);
  enum sp_transport (*get_port_transport)(const struct sp_port *port);
  enum sp_return (*get_port_usb_bus_address)(const struct sp_port *port, int *usb_bus,
                                             int *usb_address);
  enum sp_return (*get_port_usb_vid_pid)(const struct sp_port *port, int *usb_vid,
                                         int *usb_pid);
  char *(*get_port_usb_manufacturer)(const struct sp_port *port);
  char *(*get_port_usb_product)(const struct sp_port *port);
  char *(*get_port_usb_serial)(const struct sp_port *port);
  char *(*get_port_bluetooth_address)(const struct sp_port *port);

  enum sp_return (*open)(struct sp_port *port, enum sp_mode flags);
  enum sp_return (*close)(struct sp_port *port);
  enum sp_return (*set_baudrate)(struct sp_port *port, int baudrate);
  enum sp_return (*set_bits)(struct sp_port *port, int bits);
  enum sp_return (*set_parity)(struct sp_port *port, enum sp_parity parity);
  enum sp_return (*set_stopbits)(struct sp_port *port, int stopbits);
  enum sp_return (*set_flowcontrol)(struct sp_port *port, enum sp_flowcontrol flowcontrol);
  enum sp_return (*blocking_read)(struct sp_port *port, void *buf, size_t count,
                                  unsigned int timeout_ms);
  enum sp_return (*blocking_write)(struct sp_port *port, const void *buf, size_t count,
                                   unsigned int timeout_ms);
  enum sp_return (*nonblocking_read)(struct sp_port *port, void *buf, size_t count);
  enum sp_return (*nonblocking_write)(struct sp_port *port, const void *buf, size_t count);

  enum sp_return (*new_event_set)(struct sp_event_set **result_ptr);
  void (*free_event_set)(struct sp_event_set *event_set);
  enum sp_return (*add_port_events)(struct sp_event_set *event_set,
                                    const struct sp_port *port, enum sp_event mask);
  enum sp_return (*wait)(struct sp_event_set *event_set, unsigned int timeout_ms);
};

static const turbo_serial_backend_ops_t libserialport_backend_ops = {
    sp_list_ports,
    sp_get_port_by_name,
    sp_free_port,
    sp_free_port_list,
    sp_get_port_name,
    sp_get_port_description,
    sp_get_port_transport,
    sp_get_port_usb_bus_address,
    sp_get_port_usb_vid_pid,
    sp_get_port_usb_manufacturer,
    sp_get_port_usb_product,
    sp_get_port_usb_serial,
    sp_get_port_bluetooth_address,
    sp_open,
    sp_close,
    sp_set_baudrate,
    sp_set_bits,
    sp_set_parity,
    sp_set_stopbits,
    sp_set_flowcontrol,
    sp_blocking_read,
    sp_blocking_write,
    sp_nonblocking_read,
    sp_nonblocking_write,
    sp_new_event_set,
    sp_free_event_set,
    sp_add_port_events,
    sp_wait,
};

static const turbo_serial_backend_ops_t *default_backend_ops(void) {
  return &libserialport_backend_ops;
}

struct turbo_serial_t {
  const turbo_serial_backend_ops_t *ops;
  turbo_serial_base_t base;
  turbo_serial_port_t port;
  turbo_serial_rings_t rings;
  turbo_serial_async_t async;
  turbo_serial_error_t error;
};

static int is_power_of_two(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static size_t min_size(size_t a, size_t b) {
  return a < b ? a : b;
}

static turbo_serial_size_result_t make_size_result(turbo_serial_result_t code, size_t count) {
  turbo_serial_size_result_t result;
  result.code = code;
  result.count = count;
  return result;
}

static char *copy_string(const char *value) {
  size_t len;
  char *copy;

  if (!value) return NULL;

  len = strlen(value) + 1;
  copy = (char *)malloc(len);
  if (!copy) return NULL;

  memcpy(copy, value, len);
  return copy;
}

static turbo_serial_result_t publish_size_result(turbo_serial_size_result_t result,
                                                 size_t *count_out) {
  if (count_out) *count_out = result.count;
  return result.code;
}

static turbo_serial_size_result_t write_ring(ring_spsc_t *ring, const void *buf, size_t count) {
  const uint8_t *src = (const uint8_t *)buf;
  size_t total = 0;

  if (!ring || (!buf && count != 0)) return make_size_result(TURBO_SERIAL_INVALID_VALUE, 0);

  while (total < count) {
    size_t chunk = min_size(count - total, ring->size - 1);
    uint8_t *dst = NULL;

    if (chunk == 0) break;

    dst = ring_spsc_write_acquire(ring, chunk);
    if (!dst) {
      const size_t available = ring_spsc_write_available(ring);
      if (available == 0) break;

      chunk = min_size(chunk, available);
      dst = ring_spsc_write_acquire(ring, chunk);
      if (!dst) {
        chunk = 1;
        dst = ring_spsc_write_acquire(ring, chunk);
      }
    }

    if (!dst) break;

    memcpy(dst, src + total, chunk);
    ring_spsc_write_release(ring, chunk);
    total += chunk;
  }

  return make_size_result(total == 0 && count != 0 ? TURBO_SERIAL_WOULD_BLOCK : TURBO_SERIAL_OK,
                          total);
}

static turbo_serial_size_result_t read_ring(ring_spsc_t *ring, void *buf, size_t count) {
  uint8_t *dst = (uint8_t *)buf;
  size_t total = 0;

  if (!ring || (!buf && count != 0)) return make_size_result(TURBO_SERIAL_INVALID_VALUE, 0);

  while (total < count) {
    size_t available = 0;
    uint8_t *src = ring_spsc_read_acquire(ring, &available);
    size_t chunk = 0;

    if (!src || available == 0) break;

    chunk = min_size(count - total, available);
    memcpy(dst + total, src, chunk);
    ring_spsc_read_release(ring, chunk);
    total += chunk;
  }

  return make_size_result(total == 0 && count != 0 ? TURBO_SERIAL_WOULD_BLOCK : TURBO_SERIAL_OK,
                          total);
}

static void set_last_error(turbo_serial_t *serial, turbo_serial_result_t result) {
  if (serial && result != TURBO_SERIAL_OK) {
    atomic_store(&serial->error.last_error, (int)result);
  }
}

static turbo_serial_result_t result_from_sp(enum sp_return result) {
  if (result >= 0) return TURBO_SERIAL_OK;

  switch (result) {
  case SP_ERR_ARG:
    return TURBO_SERIAL_INVALID_VALUE;
  case SP_ERR_FAIL:
    return TURBO_SERIAL_IO_FAILED;
  case SP_ERR_MEM:
    return TURBO_SERIAL_NO_MEMORY;
  case SP_ERR_SUPP:
    return TURBO_SERIAL_NOT_SUPPORTED;
  case SP_ERR_TIMEOUT:
    return TURBO_SERIAL_WOULD_BLOCK;
  default:
    return TURBO_SERIAL_IO_FAILED;
  }
}

static enum sp_mode mode_to_sp(turbo_serial_mode_t mode) {
  switch (mode) {
  case TURBO_SERIAL_MODE_READ:
    return SP_MODE_READ;
  case TURBO_SERIAL_MODE_WRITE:
    return SP_MODE_WRITE;
  case TURBO_SERIAL_MODE_READ_WRITE:
    return SP_MODE_READ_WRITE;
  default:
    return 0;
  }
}

static enum sp_parity parity_to_sp(turbo_serial_parity_t parity) {
  switch (parity) {
  case TURBO_SERIAL_PARITY_INVALID:
    return SP_PARITY_INVALID;
  case TURBO_SERIAL_PARITY_NONE:
    return SP_PARITY_NONE;
  case TURBO_SERIAL_PARITY_ODD:
    return SP_PARITY_ODD;
  case TURBO_SERIAL_PARITY_EVEN:
    return SP_PARITY_EVEN;
  case TURBO_SERIAL_PARITY_MARK:
    return SP_PARITY_MARK;
  case TURBO_SERIAL_PARITY_SPACE:
    return SP_PARITY_SPACE;
  default:
    return SP_PARITY_INVALID;
  }
}

static enum sp_flowcontrol flowcontrol_to_sp(turbo_serial_flowcontrol_t flowcontrol) {
  switch (flowcontrol) {
  case TURBO_SERIAL_FLOWCONTROL_NONE:
    return SP_FLOWCONTROL_NONE;
  case TURBO_SERIAL_FLOWCONTROL_XONXOFF:
    return SP_FLOWCONTROL_XONXOFF;
  case TURBO_SERIAL_FLOWCONTROL_RTSCTS:
    return SP_FLOWCONTROL_RTSCTS;
  case TURBO_SERIAL_FLOWCONTROL_DTRDSR:
    return SP_FLOWCONTROL_DTRDSR;
  default:
    return SP_FLOWCONTROL_NONE;
  }
}

static enum sp_event events_to_sp(unsigned int events) {
  enum sp_event mask = 0;

  if (events & TURBO_SERIAL_EVENT_RX_READY) mask |= SP_EVENT_RX_READY;
  if (events & TURBO_SERIAL_EVENT_TX_READY) mask |= SP_EVENT_TX_READY;
  if (events & TURBO_SERIAL_EVENT_ERROR) mask |= SP_EVENT_ERROR;
  return mask;
}

static turbo_serial_transport_t transport_from_sp(enum sp_transport transport) {
  switch (transport) {
  case SP_TRANSPORT_NATIVE:
    return TURBO_SERIAL_TRANSPORT_NATIVE;
  case SP_TRANSPORT_USB:
    return TURBO_SERIAL_TRANSPORT_USB;
  case SP_TRANSPORT_BLUETOOTH:
    return TURBO_SERIAL_TRANSPORT_BLUETOOTH;
  default:
    return TURBO_SERIAL_TRANSPORT_UNKNOWN;
  }
}

static void clear_port_info_storage(turbo_serial_port_info_storage_t *storage) {
  if (!storage) return;

  free(storage->name);
  free(storage->description);
  free(storage->usb_manufacturer);
  free(storage->usb_product);
  free(storage->usb_serial);
  free(storage->bluetooth_address);
  memset(storage, 0, sizeof(*storage));
}

static turbo_serial_result_t fill_port_info(turbo_serial_port_info_storage_t *storage,
                                            const turbo_serial_backend_ops_t *ops,
                                            const struct sp_port *port) {
  enum sp_return sp_result;
  const char *value;

  if (!storage || !ops || !port) return TURBO_SERIAL_INVALID_VALUE;
  memset(storage, 0, sizeof(*storage));

  storage->name = copy_string(ops->get_port_name(port));
  if (!storage->name) return TURBO_SERIAL_NO_MEMORY;
  storage->view.name = storage->name;

  value = ops->get_port_description(port);
  storage->description = copy_string(value);
  if (value && !storage->description) return TURBO_SERIAL_NO_MEMORY;
  storage->view.description = storage->description;
  storage->view.transport = transport_from_sp(ops->get_port_transport(port));

  sp_result = ops->get_port_usb_bus_address(port, &storage->view.usb_bus,
                                            &storage->view.usb_address);
  storage->view.has_usb_bus_address = sp_result == SP_OK ? 1 : 0;

  sp_result = ops->get_port_usb_vid_pid(port, &storage->view.usb_vid,
                                        &storage->view.usb_pid);
  storage->view.has_usb_vid_pid = sp_result == SP_OK ? 1 : 0;

  value = ops->get_port_usb_manufacturer(port);
  storage->usb_manufacturer = copy_string(value);
  if (value && !storage->usb_manufacturer) return TURBO_SERIAL_NO_MEMORY;

  value = ops->get_port_usb_product(port);
  storage->usb_product = copy_string(value);
  if (value && !storage->usb_product) return TURBO_SERIAL_NO_MEMORY;

  value = ops->get_port_usb_serial(port);
  storage->usb_serial = copy_string(value);
  if (value && !storage->usb_serial) return TURBO_SERIAL_NO_MEMORY;

  value = ops->get_port_bluetooth_address(port);
  storage->bluetooth_address = copy_string(value);
  if (value && !storage->bluetooth_address) return TURBO_SERIAL_NO_MEMORY;

  storage->view.usb_manufacturer = storage->usb_manufacturer;
  storage->view.usb_product = storage->usb_product;
  storage->view.usb_serial = storage->usb_serial;
  storage->view.bluetooth_address = storage->bluetooth_address;
  return TURBO_SERIAL_OK;
}

static void wake_worker(turbo_serial_t *serial) {
  if (!serial) return;

  atomic_store(&serial->async.wake_requested, true);
  turbo_mutex_lock(&serial->async.worker_mutex);
  turbo_cond_signal(&serial->async.worker_cond);
  turbo_mutex_unlock(&serial->async.worker_mutex);
}

static void pump_timer_cb(turbo_timer_t *timer) {
  turbo_serial_t *serial = (turbo_serial_t *)turbo_timer_get_data(timer);
  wake_worker(serial);
}

static int pump_rx_once(turbo_serial_t *serial) {
  size_t write_available;
  size_t chunk;
  uint8_t *dst;
  enum sp_return sp_result;

  if (!serial || !serial->port.port) return 0;

  write_available = ring_spsc_write_available(&serial->rings.rx_ring);
  if (write_available == 0) return 0;

  chunk = min_size(write_available, serial->base.config.io_chunk_size);
  dst = ring_spsc_write_acquire(&serial->rings.rx_ring, chunk);
  if (!dst) {
    chunk = 1;
    dst = ring_spsc_write_acquire(&serial->rings.rx_ring, chunk);
    if (!dst) return 0;
  }

  sp_result = serial->ops->nonblocking_read(serial->port.port, dst, chunk);
  if (sp_result < 0) {
    set_last_error(serial, result_from_sp(sp_result));
    return 0;
  }
  if (sp_result == 0) return 0;

  ring_spsc_write_release(&serial->rings.rx_ring, (size_t)sp_result);
  return 1;
}

static int pump_tx_once(turbo_serial_t *serial) {
  size_t available = 0;
  size_t chunk;
  uint8_t *src;
  enum sp_return sp_result;

  if (!serial || !serial->port.port) return 0;

  src = ring_spsc_read_acquire(&serial->rings.tx_ring, &available);
  if (!src || available == 0) return 0;

  chunk = min_size(available, serial->base.config.io_chunk_size);
  sp_result = serial->ops->nonblocking_write(serial->port.port, src, chunk);
  if (sp_result < 0) {
    set_last_error(serial, result_from_sp(sp_result));
    return 0;
  }
  if (sp_result == 0) return 0;

  ring_spsc_read_release(&serial->rings.tx_ring, (size_t)sp_result);
  return 1;
}

static void pump_worker_main(void *arg) {
  turbo_serial_t *serial = (turbo_serial_t *)arg;

  while (atomic_load(&serial->async.async_running)) {
    int made_progress = 0;

    made_progress |= pump_rx_once(serial);
    made_progress |= pump_tx_once(serial);

    if (made_progress) {
      turbo_thread_yield();
      continue;
    }

    turbo_mutex_lock(&serial->async.worker_mutex);
    if (atomic_load(&serial->async.async_running) &&
        !atomic_exchange(&serial->async.wake_requested, false)) {
      turbo_cond_wait(&serial->async.worker_cond, &serial->async.worker_mutex);
    }
    turbo_mutex_unlock(&serial->async.worker_mutex);
  }
}

void turbo_serial_config_default(turbo_serial_config_t *config) {
  if (!config) return;

  config->baudrate = 115200;
  config->bits = 8;
  config->parity = TURBO_SERIAL_PARITY_NONE;
  config->stopbits = 1;
  config->flowcontrol = TURBO_SERIAL_FLOWCONTROL_NONE;
  config->rx_buffer_size = 4096;
  config->tx_buffer_size = 4096;
  config->io_chunk_size = 256;
  config->poll_interval_ms = 10;
}

const char *turbo_serial_result_name(turbo_serial_result_t result) {
  switch (result) {
  case TURBO_SERIAL_OK:
    return "ok";
  case TURBO_SERIAL_INVALID_VALUE:
    return "invalid value";
  case TURBO_SERIAL_INVALID_STATE:
    return "invalid state";
  case TURBO_SERIAL_IO_FAILED:
    return "io failed";
  case TURBO_SERIAL_NO_MEMORY:
    return "no memory";
  case TURBO_SERIAL_NOT_SUPPORTED:
    return "not supported";
  case TURBO_SERIAL_WOULD_BLOCK:
    return "would block";
  default:
    return "unknown";
  }
}

turbo_serial_result_t turbo_serial_list_ports(turbo_serial_port_list_t **ports) {
  const turbo_serial_backend_ops_t *ops = default_backend_ops();
  struct sp_port **sp_ports = NULL;
  struct sp_port **cursor;
  turbo_serial_port_list_t *list = NULL;
  enum sp_return sp_result;
  turbo_serial_result_t result = TURBO_SERIAL_OK;
  size_t count = 0;
  size_t i;

  if (!ports) return TURBO_SERIAL_INVALID_VALUE;
  *ports = NULL;

  sp_result = ops->list_ports(&sp_ports);
  if (sp_result < 0) return result_from_sp(sp_result);

  for (cursor = sp_ports; cursor && *cursor; ++cursor) {
    ++count;
  }

  list = (turbo_serial_port_list_t *)calloc(1, sizeof(*list));
  if (!list) {
    result = TURBO_SERIAL_NO_MEMORY;
    goto cleanup;
  }

  if (count > 0) {
    list->items = (turbo_serial_port_info_storage_t *)calloc(count, sizeof(*list->items));
    if (!list->items) {
      result = TURBO_SERIAL_NO_MEMORY;
      goto cleanup;
    }
  }

  list->count = count;
  for (i = 0; i < count; ++i) {
    result = fill_port_info(&list->items[i], ops, sp_ports[i]);
    if (result != TURBO_SERIAL_OK) goto cleanup;
  }

  *ports = list;
  ops->free_port_list(sp_ports);
  return TURBO_SERIAL_OK;

cleanup:
  ops->free_port_list(sp_ports);
  turbo_serial_port_list_destroy(list);
  return result;
}

turbo_serial_result_t turbo_serial_port_info_by_name(const char *port_name,
                                                     turbo_serial_port_list_t **ports,
                                                     const turbo_serial_port_info_t **info) {
  const turbo_serial_backend_ops_t *ops = default_backend_ops();
  struct sp_port *port = NULL;
  turbo_serial_port_list_t *list = NULL;
  enum sp_return sp_result;
  turbo_serial_result_t result;

  if (!ports) return TURBO_SERIAL_INVALID_VALUE;
  *ports = NULL;
  if (info) *info = NULL;
  if (!port_name) return TURBO_SERIAL_INVALID_VALUE;

  sp_result = ops->get_port_by_name(port_name, &port);
  if (sp_result < 0) return result_from_sp(sp_result);

  list = (turbo_serial_port_list_t *)calloc(1, sizeof(*list));
  if (!list) {
    result = TURBO_SERIAL_NO_MEMORY;
    goto cleanup;
  }

  list->items = (turbo_serial_port_info_storage_t *)calloc(1, sizeof(*list->items));
  if (!list->items) {
    result = TURBO_SERIAL_NO_MEMORY;
    goto cleanup;
  }
  list->count = 1;

  result = fill_port_info(&list->items[0], ops, port);
  if (result != TURBO_SERIAL_OK) goto cleanup;

  *ports = list;
  if (info) *info = &list->items[0].view;
  ops->free_port(port);
  return TURBO_SERIAL_OK;

cleanup:
  ops->free_port(port);
  turbo_serial_port_list_destroy(list);
  return result;
}

void turbo_serial_port_list_destroy(turbo_serial_port_list_t *ports) {
  size_t i;

  if (!ports) return;

  for (i = 0; i < ports->count; ++i) {
    clear_port_info_storage(&ports->items[i]);
  }
  free(ports->items);
  free(ports);
}

size_t turbo_serial_port_list_count(const turbo_serial_port_list_t *ports) {
  return ports ? ports->count : 0;
}

const turbo_serial_port_info_t *turbo_serial_port_list_get(const turbo_serial_port_list_t *ports,
                                                           size_t index) {
  if (!ports || index >= ports->count) return NULL;
  return &ports->items[index].view;
}

turbo_serial_result_t turbo_serial_create(turbo_serial_t **serial,
                                          const turbo_serial_config_t *config) {
  turbo_serial_config_t actual;
  turbo_serial_t *instance = NULL;
  turbo_serial_result_t result = TURBO_SERIAL_OK;

  if (!serial) return TURBO_SERIAL_INVALID_VALUE;
  *serial = NULL;

  if (config) actual = *config;
  else turbo_serial_config_default(&actual);

  if (!is_power_of_two(actual.rx_buffer_size) || !is_power_of_two(actual.tx_buffer_size) ||
      actual.rx_buffer_size < 2 || actual.tx_buffer_size < 2 || actual.io_chunk_size == 0 ||
      actual.poll_interval_ms == 0) {
    return TURBO_SERIAL_INVALID_VALUE;
  }

  instance = (turbo_serial_t *)calloc(1, sizeof(*instance));
  if (!instance) return TURBO_SERIAL_NO_MEMORY;

  instance->ops = default_backend_ops();
  turbo_mutex_init(&instance->async.worker_mutex);
  turbo_cond_init(&instance->async.worker_cond);
  atomic_init(&instance->async.async_running, false);
  atomic_init(&instance->async.wake_requested, false);
  atomic_init(&instance->error.last_error, (int)TURBO_SERIAL_OK);

  instance->rings.rx_storage = (uint8_t *)malloc(actual.rx_buffer_size);
  instance->rings.tx_storage = (uint8_t *)malloc(actual.tx_buffer_size);
  if (!instance->rings.rx_storage || !instance->rings.tx_storage) {
    result = TURBO_SERIAL_NO_MEMORY;
    goto cleanup;
  }

  if (!ring_spsc_init(&instance->rings.rx_ring, instance->rings.rx_storage,
                      actual.rx_buffer_size) ||
      !ring_spsc_init(&instance->rings.tx_ring, instance->rings.tx_storage,
                      actual.tx_buffer_size)) {
    result = TURBO_SERIAL_INVALID_VALUE;
    goto cleanup;
  }

  instance->base.config = actual;
  *serial = instance;
  return TURBO_SERIAL_OK;

cleanup:
  turbo_serial_destroy(instance);
  return result;
}

void turbo_serial_destroy(turbo_serial_t *serial) {
  if (!serial) return;

  (void)turbo_serial_close(serial);
  if (serial->async.pump_timer) {
    turbo_timer_destroy(serial->async.pump_timer);
    serial->async.pump_timer = NULL;
  }
  turbo_cond_destroy(&serial->async.worker_cond);
  turbo_mutex_destroy(&serial->async.worker_mutex);
  free(serial->rings.rx_storage);
  free(serial->rings.tx_storage);
  free(serial);
}

turbo_serial_result_t turbo_serial_open(turbo_serial_t *serial, const char *port_name,
                                        turbo_serial_mode_t mode) {
  struct sp_port *port = NULL;
  enum sp_return sp_result;
  enum sp_mode sp_mode;
  turbo_serial_result_t result;
  int opened = 0;

  if (!serial || !port_name) return TURBO_SERIAL_INVALID_VALUE;
  if (serial->port.port) return TURBO_SERIAL_INVALID_STATE;
  sp_mode = mode_to_sp(mode);
  if (sp_mode == 0) return TURBO_SERIAL_INVALID_VALUE;

  sp_result = serial->ops->get_port_by_name(port_name, &port);
  if (sp_result < 0) return result_from_sp(sp_result);

  sp_result = serial->ops->open(port, sp_mode);
  if (sp_result < 0) {
    result = result_from_sp(sp_result);
    goto cleanup;
  }
  opened = 1;

  sp_result = serial->ops->set_baudrate(port, serial->base.config.baudrate);
  if (sp_result == SP_OK) sp_result = serial->ops->set_bits(port, serial->base.config.bits);
  if (sp_result == SP_OK) {
    sp_result = serial->ops->set_parity(port, parity_to_sp(serial->base.config.parity));
  }
  if (sp_result == SP_OK) sp_result = serial->ops->set_stopbits(port, serial->base.config.stopbits);
  if (sp_result == SP_OK) {
    sp_result = serial->ops->set_flowcontrol(port,
                                             flowcontrol_to_sp(serial->base.config.flowcontrol));
  }
  if (sp_result < 0) {
    result = result_from_sp(sp_result);
    goto cleanup;
  }

  serial->port.port = port;
  return TURBO_SERIAL_OK;

cleanup:
  if (port) {
    if (opened) (void)serial->ops->close(port);
    serial->ops->free_port(port);
  }
  return result;
}

turbo_serial_result_t turbo_serial_close(turbo_serial_t *serial) {
  enum sp_return sp_result;

  if (!serial) return TURBO_SERIAL_INVALID_VALUE;
  if (!serial->port.port) return TURBO_SERIAL_OK;

  (void)turbo_serial_stop_async(serial);

  sp_result = serial->ops->close(serial->port.port);
  serial->ops->free_port(serial->port.port);
  serial->port.port = NULL;

  return result_from_sp(sp_result);
}

turbo_serial_result_t turbo_serial_read(turbo_serial_t *serial, void *buf, size_t count,
                                        unsigned int timeout_ms, size_t *bytes_read) {
  enum sp_return sp_result;

  if (bytes_read) *bytes_read = 0;
  if (!serial || (!buf && count != 0)) return TURBO_SERIAL_INVALID_VALUE;
  if (!serial->port.port) return TURBO_SERIAL_INVALID_STATE;
  if (atomic_load(&serial->async.async_running)) return TURBO_SERIAL_INVALID_STATE;

  sp_result = serial->ops->blocking_read(serial->port.port, buf, count, timeout_ms);
  if (sp_result < 0) return result_from_sp(sp_result);

  if (bytes_read) *bytes_read = (size_t)sp_result;
  return TURBO_SERIAL_OK;
}

turbo_serial_result_t turbo_serial_write(turbo_serial_t *serial, const void *buf, size_t count,
                                         unsigned int timeout_ms, size_t *bytes_written) {
  enum sp_return sp_result;

  if (bytes_written) *bytes_written = 0;
  if (!serial || (!buf && count != 0)) return TURBO_SERIAL_INVALID_VALUE;
  if (!serial->port.port) return TURBO_SERIAL_INVALID_STATE;
  if (atomic_load(&serial->async.async_running)) return TURBO_SERIAL_INVALID_STATE;

  sp_result = serial->ops->blocking_write(serial->port.port, buf, count, timeout_ms);
  if (sp_result < 0) return result_from_sp(sp_result);

  if (bytes_written) *bytes_written = (size_t)sp_result;
  return TURBO_SERIAL_OK;
}

turbo_serial_result_t turbo_serial_start_async(turbo_serial_t *serial) {
  int thread_result;
  int timer_result;

  if (!serial) return TURBO_SERIAL_INVALID_VALUE;
  if (!serial->port.port) return TURBO_SERIAL_INVALID_STATE;
  if (atomic_load(&serial->async.async_running)) return TURBO_SERIAL_OK;

  if (!serial->async.pump_timer) {
    serial->async.pump_timer = turbo_timer_create(NULL);
    if (!serial->async.pump_timer) return TURBO_SERIAL_NO_MEMORY;
    turbo_timer_set_data(serial->async.pump_timer, serial);
  }

  atomic_store(&serial->error.last_error, (int)TURBO_SERIAL_OK);
  atomic_store(&serial->async.wake_requested, true);
  atomic_store(&serial->async.async_running, true);

  thread_result = turbo_thread_create(&serial->async.worker_thread, pump_worker_main, serial);
  if (thread_result != 0) {
    atomic_store(&serial->async.async_running, false);
    return TURBO_SERIAL_IO_FAILED;
  }

  timer_result = turbo_timer_start(serial->async.pump_timer, pump_timer_cb,
                                   serial->base.config.poll_interval_ms,
                                   serial->base.config.poll_interval_ms);
  if (timer_result != 0) {
    atomic_store(&serial->async.async_running, false);
    wake_worker(serial);
    (void)turbo_thread_join(&serial->async.worker_thread);
    return TURBO_SERIAL_IO_FAILED;
  }

  wake_worker(serial);
  return TURBO_SERIAL_OK;
}

turbo_serial_result_t turbo_serial_stop_async(turbo_serial_t *serial) {
  if (!serial) return TURBO_SERIAL_INVALID_VALUE;
  if (!atomic_load(&serial->async.async_running)) return TURBO_SERIAL_OK;

  atomic_store(&serial->async.async_running, false);
  if (serial->async.pump_timer) {
    (void)turbo_timer_stop(serial->async.pump_timer);
  }
  wake_worker(serial);
  if (serial->async.worker_thread) {
    (void)turbo_thread_join(&serial->async.worker_thread);
  }
  return TURBO_SERIAL_OK;
}

int turbo_serial_async_running(const turbo_serial_t *serial) {
  if (!serial) return 0;
  return atomic_load(&serial->async.async_running) ? 1 : 0;
}

turbo_serial_result_t turbo_serial_last_error(const turbo_serial_t *serial) {
  if (!serial) return TURBO_SERIAL_INVALID_VALUE;
  return (turbo_serial_result_t)atomic_load(&serial->error.last_error);
}

turbo_serial_result_t turbo_serial_event_set_create(turbo_serial_event_set_t **event_set) {
  const turbo_serial_backend_ops_t *ops = default_backend_ops();
  turbo_serial_event_set_t *instance = NULL;
  enum sp_return sp_result;

  if (!event_set) return TURBO_SERIAL_INVALID_VALUE;
  *event_set = NULL;

  instance = (turbo_serial_event_set_t *)calloc(1, sizeof(*instance));
  if (!instance) return TURBO_SERIAL_NO_MEMORY;

  instance->ops = ops;
  sp_result = ops->new_event_set(&instance->set);
  if (sp_result < 0) {
    free(instance);
    return result_from_sp(sp_result);
  }

  *event_set = instance;
  return TURBO_SERIAL_OK;
}

void turbo_serial_event_set_destroy(turbo_serial_event_set_t *event_set) {
  if (!event_set) return;

  if (event_set->set) event_set->ops->free_event_set(event_set->set);
  free(event_set);
}

turbo_serial_result_t turbo_serial_event_set_add(turbo_serial_event_set_t *event_set,
                                                 const turbo_serial_t *serial,
                                                 unsigned int events) {
  enum sp_event sp_events;
  enum sp_return sp_result;

  if (!event_set || !event_set->set || !event_set->ops || !serial || !serial->ops) {
    return TURBO_SERIAL_INVALID_VALUE;
  }
  if (event_set->ops != serial->ops) return TURBO_SERIAL_INVALID_STATE;
  if ((events & ~(TURBO_SERIAL_EVENT_RX_READY | TURBO_SERIAL_EVENT_TX_READY |
                  TURBO_SERIAL_EVENT_ERROR)) != 0) {
    return TURBO_SERIAL_INVALID_VALUE;
  }
  if (!serial->port.port) return TURBO_SERIAL_INVALID_STATE;

  sp_events = events_to_sp(events);
  if (sp_events == 0) return TURBO_SERIAL_OK;

  sp_result = event_set->ops->add_port_events(event_set->set, serial->port.port, sp_events);
  if (sp_result == SP_OK) event_set->count++;
  return result_from_sp(sp_result);
}

turbo_serial_result_t turbo_serial_event_wait(turbo_serial_event_set_t *event_set,
                                              unsigned int timeout_ms) {
  enum sp_return sp_result;

  if (!event_set || !event_set->set) return TURBO_SERIAL_INVALID_VALUE;
  if (event_set->count == 0) return TURBO_SERIAL_INVALID_STATE;

  sp_result = event_set->ops->wait(event_set->set, timeout_ms);
  return result_from_sp(sp_result);
}

size_t turbo_serial_rx_available(const turbo_serial_t *serial) {
  if (!serial) return 0;
  return ring_spsc_read_available(&serial->rings.rx_ring);
}

size_t turbo_serial_tx_available(const turbo_serial_t *serial) {
  if (!serial) return 0;
  return ring_spsc_read_available(&serial->rings.tx_ring);
}

size_t turbo_serial_rx_capacity(const turbo_serial_t *serial) {
  if (!serial) return 0;
  return serial->rings.rx_ring.size - 1;
}

size_t turbo_serial_tx_capacity(const turbo_serial_t *serial) {
  if (!serial) return 0;
  return serial->rings.tx_ring.size - 1;
}

turbo_serial_result_t turbo_serial_buffer_rx(turbo_serial_t *serial, const void *buf,
                                             size_t count, size_t *bytes_buffered) {
  turbo_serial_size_result_t result;

  if (bytes_buffered) *bytes_buffered = 0;
  if (!serial) return TURBO_SERIAL_INVALID_VALUE;
  if (atomic_load(&serial->async.async_running)) return TURBO_SERIAL_INVALID_STATE;
  result = write_ring(&serial->rings.rx_ring, buf, count);
  return publish_size_result(result, bytes_buffered);
}

turbo_serial_result_t turbo_serial_read_buffered(turbo_serial_t *serial, void *buf,
                                                 size_t count, size_t *bytes_read) {
  turbo_serial_size_result_t result;

  if (bytes_read) *bytes_read = 0;
  if (!serial) return TURBO_SERIAL_INVALID_VALUE;
  result = read_ring(&serial->rings.rx_ring, buf, count);
  return publish_size_result(result, bytes_read);
}

turbo_serial_result_t turbo_serial_write_buffered(turbo_serial_t *serial, const void *buf,
                                                  size_t count, size_t *bytes_buffered) {
  turbo_serial_size_result_t result;

  if (bytes_buffered) *bytes_buffered = 0;
  if (!serial) return TURBO_SERIAL_INVALID_VALUE;
  result = write_ring(&serial->rings.tx_ring, buf, count);
  if (result.code == TURBO_SERIAL_OK && result.count > 0) {
    wake_worker(serial);
  }
  return publish_size_result(result, bytes_buffered);
}

turbo_serial_result_t turbo_serial_drain_tx_buffer(turbo_serial_t *serial, void *buf,
                                                   size_t count, size_t *bytes_read) {
  turbo_serial_size_result_t result;

  if (bytes_read) *bytes_read = 0;
  if (!serial) return TURBO_SERIAL_INVALID_VALUE;
  if (atomic_load(&serial->async.async_running)) return TURBO_SERIAL_INVALID_STATE;
  result = read_ring(&serial->rings.tx_ring, buf, count);
  return publish_size_result(result, bytes_read);
}
