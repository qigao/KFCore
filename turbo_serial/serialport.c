#include "serialport.h"

static const struct std_baudrate std_baudrates[] = {
#ifdef _WIN32
    /*
     * The baudrates 50/75/134/150/200/1800/230400/460800 do not seem to
     * have documented CBR_* macros.
     */
    BAUD(110),   BAUD(300),   BAUD(600),   BAUD(1200),  BAUD(2400),   BAUD(4800),   BAUD(9600),
    BAUD(14400), BAUD(19200), BAUD(38400), BAUD(57600), BAUD(115200), BAUD(128000), BAUD(256000),
#else
    BAUD(50),     BAUD(75),    BAUD(110),   BAUD(134),   BAUD(150),    BAUD(200),
    BAUD(300),    BAUD(600),   BAUD(1200),  BAUD(1800),  BAUD(2400),   BAUD(4800),
    BAUD(9600),   BAUD(19200), BAUD(38400), BAUD(57600), BAUD(115200), BAUD(230400),
  #if !defined(__APPLE__) && !defined(__OpenBSD__)
    BAUD(460800),
  #endif
#endif
};

#define NUM_STD_BAUDRATES ARRAY_SIZE(std_baudrates)

static enum sp_return get_config(struct sp_port *port, struct port_data *data,
                                 struct sp_port_config *config);

static enum sp_return set_config(struct sp_port *port, struct port_data *data,
                                 const struct sp_port_config *config);

enum sp_return sp_get_port_by_name(const char *portname, struct sp_port **port_ptr) {
  struct sp_port *port;
#ifndef NO_PORT_METADATA
  enum sp_return ret;
#endif
  size_t len;
  if (!port_ptr) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null result pointer");

  *port_ptr = NULL;

  if (!portname) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null port name");

  TLOG_DEBUG("Building structure for port {}", portname);

#if !defined(_WIN32) && defined(HAVE_REALPATH)
  /*
   * get_port_details() below tries to be too smart and figure out
   * some transport properties from the port name which breaks with
   * symlinks. Therefore we canonicalize the portname first.
   */
  char pathbuf[PATH_MAX + 1];
  char *res = realpath(portname, pathbuf);
  if (!res) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Could not retrieve realpath behind port name");

  portname = pathbuf;
#endif

  if (!(port = malloc(sizeof(struct sp_port))))
    return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "Port structure malloc failed");

  len = strlen(portname) + 1;

  if (!(port->name = malloc(len))) {
    free(port);
    return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "Port name malloc failed");
  }

  memcpy(port->name, portname, len);

#ifdef _WIN32
  port->usb_path = NULL;
  port->hdl = INVALID_HANDLE_VALUE;
  port->write_buf = NULL;
  port->write_buf_size = 0;
#else
  port->fd = -1;
#endif

  port->description = NULL;
  port->transport = SP_TRANSPORT_NATIVE;
  port->usb_bus = -1;
  port->usb_address = -1;
  port->usb_vid = -1;
  port->usb_pid = -1;
  port->usb_manufacturer = NULL;
  port->usb_product = NULL;
  port->usb_serial = NULL;
  port->bluetooth_address = NULL;

#ifndef NO_PORT_METADATA
  if ((ret = get_port_details(port)) != SP_OK) {
    sp_free_port(port);
    return ret;
  }
#endif

  *port_ptr = port;

  return SP_OK;
}

char *sp_get_port_name(const struct sp_port *port) {
  if (!port) return NULL;

  return port->name;
}

char *sp_get_port_description(const struct sp_port *port) {
  if (!port || !port->description) return NULL;

  return port->description;
}

enum sp_transport sp_get_port_transport(const struct sp_port *port) {
  return port ? port->transport : SP_TRANSPORT_NATIVE;
}

enum sp_return sp_get_port_usb_bus_address(const struct sp_port *port, int *usb_bus,
                                                  int *usb_address) {
  if (!port) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null port");
  if (port->transport != SP_TRANSPORT_USB)
    return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Port does not use USB transport");
  if (port->usb_bus < 0 || port->usb_address < 0)
    return sp_error_return(__func__, SP_ERR_SUPP, "SP_ERR_SUPP", "Bus and address values are not available");

  if (usb_bus) *usb_bus = port->usb_bus;
  if (usb_address) *usb_address = port->usb_address;

  return SP_OK;
}

enum sp_return sp_get_port_usb_vid_pid(const struct sp_port *port, int *usb_vid,
                                              int *usb_pid) {
  if (!port) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null port");
  if (port->transport != SP_TRANSPORT_USB)
    return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Port does not use USB transport");
  if (port->usb_vid < 0 || port->usb_pid < 0)
    return sp_error_return(__func__, SP_ERR_SUPP, "SP_ERR_SUPP", "VID:PID values are not available");

  if (usb_vid) *usb_vid = port->usb_vid;
  if (usb_pid) *usb_pid = port->usb_pid;

  return SP_OK;
}

char *sp_get_port_usb_manufacturer(const struct sp_port *port) {
  if (!port || port->transport != SP_TRANSPORT_USB || !port->usb_manufacturer) return NULL;

  return port->usb_manufacturer;
}

char *sp_get_port_usb_product(const struct sp_port *port) {
  if (!port || port->transport != SP_TRANSPORT_USB || !port->usb_product) return NULL;

  return port->usb_product;
}

char *sp_get_port_usb_serial(const struct sp_port *port) {
  if (!port || port->transport != SP_TRANSPORT_USB || !port->usb_serial) return NULL;

  return port->usb_serial;
}

char *sp_get_port_bluetooth_address(const struct sp_port *port) {
  if (!port || port->transport != SP_TRANSPORT_BLUETOOTH || !port->bluetooth_address) return NULL;

  return port->bluetooth_address;
}

enum sp_return sp_get_port_handle(const struct sp_port *port, void *result_ptr) {
  if (!port) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null port");
  if (!result_ptr) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null result pointer");

#ifdef _WIN32
  HANDLE *handle_ptr = result_ptr;
  *handle_ptr = port->hdl;
#else
  int *fd_ptr = result_ptr;
  *fd_ptr = port->fd;
#endif

  return SP_OK;
}

enum sp_return sp_copy_port(const struct sp_port *port, struct sp_port **copy_ptr) {
  if (!copy_ptr) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null result pointer");

  *copy_ptr = NULL;

  if (!port) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null port");

  if (!port->name) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null port name");

  TLOG_DEBUG("Copying port structure");

  return sp_get_port_by_name(port->name, copy_ptr);
}

void sp_free_port(struct sp_port *port) {
  if (!port) {
    TLOG_DEBUG("Null port");
    return;
  }

  TLOG_DEBUG("Freeing port structure");

  if (port->name) free(port->name);
  if (port->description) free(port->description);
  if (port->usb_manufacturer) free(port->usb_manufacturer);
  if (port->usb_product) free(port->usb_product);
  if (port->usb_serial) free(port->usb_serial);
  if (port->bluetooth_address) free(port->bluetooth_address);
#ifdef _WIN32
  if (port->usb_path) free(port->usb_path);
  if (port->write_buf) free(port->write_buf);
#endif

  free(port);

  return;
}

struct sp_port **list_append(struct sp_port **list, const char *portname) {
  void *tmp;
  size_t count;

  for (count = 0; list[count]; count++)
    ;
  if (!(tmp = realloc(list, sizeof(struct sp_port *) * (count + 2)))) goto fail;
  list = tmp;
  if (sp_get_port_by_name(portname, &list[count]) != SP_OK) goto fail;
  list[count + 1] = NULL;
  return list;

fail:
  sp_free_port_list(list);
  return NULL;
}

enum sp_return sp_list_ports(struct sp_port ***list_ptr) {
#ifndef NO_ENUMERATION
  struct sp_port **list;
  int ret;
#endif
  if (!list_ptr) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null result pointer");

  *list_ptr = NULL;

#ifdef NO_ENUMERATION
  return sp_error_return(__func__, SP_ERR_SUPP, "SP_ERR_SUPP", "Enumeration not supported on this platform");
#else
  TLOG_DEBUG("Enumerating ports");

  if (!(list = malloc(sizeof(struct sp_port *))))
    return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "Port list malloc failed");

  list[0] = NULL;

  ret = list_ports(&list);

  if (ret == SP_OK) {
    *list_ptr = list;
  } else {
    sp_free_port_list(list);
    *list_ptr = NULL;
  }

  return sp_normalize_return(ret);
#endif
}

void sp_free_port_list(struct sp_port **list) {
  unsigned int i;
  if (!list) {
    TLOG_DEBUG("Null list");
    return;
  }

  TLOG_DEBUG("Freeing port list");

  for (i = 0; list[i]; i++)
    sp_free_port(list[i]);
  free(list);

  return;
}

static enum sp_return sp_validate_port(const struct sp_port *port, const char *function) {
  if (!port) return sp_error_return(function, SP_ERR_ARG, "SP_ERR_ARG", "Null port");
  if (!port->name) return sp_error_return(function, SP_ERR_ARG, "SP_ERR_ARG", "Null port name");
  return SP_OK;
}

#ifdef _WIN32
static enum sp_return sp_validate_open_port(const struct sp_port *port, const char *function) {
  enum sp_return ret = sp_validate_port(port, function);
  if (ret != SP_OK) return ret;
  if (port->hdl == INVALID_HANDLE_VALUE)
    return sp_error_return(function, SP_ERR_ARG, "SP_ERR_ARG", "Port not open");
  return SP_OK;
}
#else
static enum sp_return sp_validate_open_port(const struct sp_port *port, const char *function) {
  enum sp_return ret = sp_validate_port(port, function);
  if (ret != SP_OK) return ret;
  if (port->fd < 0)
    return sp_error_return(function, SP_ERR_ARG, "SP_ERR_ARG", "Port not open");
  return SP_OK;
}
#endif

#ifdef WIN32
/** To be called after port receive buffer is emptied. */
static enum sp_return restart_wait(struct sp_port *port) {
  DWORD wait_result;

  if (port->wait_running) {
    /* Check status of running wait operation. */
    if (GetOverlappedResult(port->hdl, &port->wait_ovl, &wait_result, FALSE)) {
      TLOG_DEBUG("Previous wait completed");
      port->last_wait_thread_exited = FALSE;
      port->wait_running = FALSE;
    } else if (GetLastError() == ERROR_OPERATION_ABORTED) {
      /* This error is returned if the last thread that called
       * restart_wait() has exited while WaitCommEvent() was
       * still active. In that case we don't consider that to
       * be an error. Just restart the wait procedure instead.
       */
      TLOG_DEBUG("Previous wait ended due to previous thread exiting");
      /* We need to record that the wait thread exited before
       * we called WaitCommEvent(). This is because the exit of
       * the previous thread always generates a spurious wakeup,
       * and if no data has been received in the mean time, the
       * WaitCommEvent() wouldn't be restarted a second time by
       * restart_wait_if_needed() after a read call after the
       * spurious wakeup.
       */
      port->last_wait_thread_exited = TRUE;
      port->wait_running = FALSE;
    } else if (GetLastError() == ERROR_IO_INCOMPLETE) {
      TLOG_DEBUG("Previous wait still running");
      port->last_wait_thread_exited = FALSE;
      return SP_OK;
    } else {
      return sp_fail_return(__func__, "GetOverlappedResult() failed");
    }
  }

  if (!port->wait_running) {
    /* Start new wait operation. */
    if (WaitCommEvent(port->hdl, &port->events, &port->wait_ovl)) {
      TLOG_DEBUG("New wait returned, events already pending");
    } else if (GetLastError() == ERROR_IO_PENDING) {
      TLOG_DEBUG("New wait running in background");
      port->wait_running = TRUE;
    } else {
      return sp_fail_return(__func__, "WaitCommEvent() failed");
    }
  }

  return SP_OK;
}
#endif

enum sp_return sp_open(struct sp_port *port, enum sp_mode flags) {
  struct port_data data;
  struct sp_port_config config;
  enum sp_return ret;
  ret = sp_validate_port(port, __func__);
  if (ret != SP_OK) return ret;

  if (flags > SP_MODE_READ_WRITE) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid flags");

  TLOG_DEBUG("Opening port {}", port->name);

#ifdef _WIN32
  DWORD desired_access = 0, flags_and_attributes = 0, errors;
  char *escaped_port_name;
  size_t escaped_port_name_len;
  COMSTAT status;

  /* Prefix port name with '\\.\' to work with ports above COM9. */
  escaped_port_name_len = strlen("\\\\.\\") + strlen(port->name) + 1;
  if (!(escaped_port_name = malloc(escaped_port_name_len)))
    return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "Escaped port name malloc failed");
  if ((size_t)snprintf(escaped_port_name, escaped_port_name_len, "\\\\.\\%s", port->name) >=
      escaped_port_name_len) {
    free(escaped_port_name);
    return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Escaped port name truncated");
  }

  /* Map 'flags' to the OS-specific settings. */
  flags_and_attributes = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
  if (flags & SP_MODE_READ) desired_access |= GENERIC_READ;
  if (flags & SP_MODE_WRITE) desired_access |= GENERIC_WRITE;

  port->hdl =
      CreateFileA(escaped_port_name, desired_access, 0, 0, OPEN_EXISTING, flags_and_attributes, 0);

  free(escaped_port_name);

  if (port->hdl == INVALID_HANDLE_VALUE) return sp_fail_return(__func__, "Port CreateFile() failed");

  /* All timeouts initially disabled. */
  port->timeouts.ReadIntervalTimeout = 0;
  port->timeouts.ReadTotalTimeoutMultiplier = 0;
  port->timeouts.ReadTotalTimeoutConstant = 0;
  port->timeouts.WriteTotalTimeoutMultiplier = 0;
  port->timeouts.WriteTotalTimeoutConstant = 0;

  if (SetCommTimeouts(port->hdl, &port->timeouts) == 0) {
    sp_close(port);
    return sp_fail_return(__func__, "SetCommTimeouts() failed");
  }

  /* Prepare OVERLAPPED structures. */
  #define INIT_OVERLAPPED(ovl)                                                                     \
    do {                                                                                           \
      memset(&port->ovl, 0, sizeof(port->ovl));                                                    \
      port->ovl.hEvent = INVALID_HANDLE_VALUE;                                                     \
      if ((port->ovl.hEvent = CreateEvent(NULL, TRUE, TRUE, NULL)) == INVALID_HANDLE_VALUE) {      \
        sp_close(port);                                                                            \
        return sp_fail_return(__func__, #ovl "CreateEvent() failed");                              \
      }                                                                                            \
    } while (0)

  INIT_OVERLAPPED(read_ovl);
  INIT_OVERLAPPED(write_ovl);
  INIT_OVERLAPPED(wait_ovl);

  /* Set event mask for RX and error events. */
  if (SetCommMask(port->hdl, EV_RXCHAR | EV_ERR) == 0) {
    sp_close(port);
    return sp_fail_return(__func__, "SetCommMask() failed");
  }

  port->writing = FALSE;
  port->wait_running = FALSE;

  ret = restart_wait(port);

  if (ret < 0) {
    sp_close(port);
    return sp_normalize_return(ret);
  }
#else
  int flags_local = O_NONBLOCK | O_NOCTTY | O_CLOEXEC;

  /* Map 'flags' to the OS-specific settings. */
  if ((flags & SP_MODE_READ_WRITE) == SP_MODE_READ_WRITE) flags_local |= O_RDWR;
  else if (flags & SP_MODE_READ) flags_local |= O_RDONLY;
  else if (flags & SP_MODE_WRITE) flags_local |= O_WRONLY;

  if ((port->fd = open(port->name, flags_local)) < 0) return sp_fail_return(__func__, "open() failed");

  /*
   * On POSIX in the default case the file descriptor of a serial port
   * is not opened exclusively. Therefore the settings of a port are
   * overwritten if the serial port is opened a second time. Windows
   * opens all serial ports exclusively.
   * So the idea is to open the serial ports alike in the exclusive mode.
   *
   * ioctl(*, TIOCEXCL) defines the file descriptor as exclusive. So all
   * further open calls on the serial port will fail.
   *
   * There is a race condition if two processes open the same serial
   * port. None of the processes will notice the exclusive ownership of
   * the other process because ioctl() doesn't return an error code if
   * the file descriptor is already marked as exclusive.
   * This can be solved with flock(). It returns an error if the file
   * descriptor is already locked by another process.
   */
  #ifdef HAVE_FLOCK
  if (flock(port->fd, LOCK_EX | LOCK_NB) < 0) return sp_fail_return(__func__, "flock() failed");
  #endif

  #ifdef TIOCEXCL
  /*
   * Before Linux 3.8 ioctl(*, TIOCEXCL) was not implemented and could
   * lead to EINVAL or ENOTTY.
   * These errors aren't fatal and can be ignored.
   */
  if (ioctl(port->fd, TIOCEXCL) < 0 && errno != EINVAL && errno != ENOTTY)
    return sp_fail_return(__func__, "ioctl() failed");
  #endif

#endif

  ret = get_config(port, &data, &config);

  if (ret < 0) {
    sp_close(port);
    return sp_normalize_return(ret);
  }

  /*
   * Assume a default baudrate if the OS does not provide one.
   * Cannot assign -1 here since Windows holds the baudrate in
   * the DCB and does not configure the rate individually.
   */
  if (config.baudrate == 0) {
    config.baudrate = 9600;
  }

  /* Set sane port settings. */
#ifdef _WIN32
  data.dcb.fBinary = TRUE;
  data.dcb.fDsrSensitivity = FALSE;
  data.dcb.fErrorChar = FALSE;
  data.dcb.fNull = FALSE;
  data.dcb.fAbortOnError = FALSE;
#else
  /* Turn off all fancy termios tricks, give us a raw channel. */
  data.term.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IMAXBEL);
  #ifdef IUCLC
  data.term.c_iflag &= ~IUCLC;
  #endif
  data.term.c_oflag &= ~(OPOST | ONLCR | OCRNL | ONOCR | ONLRET);
  #ifdef OLCUC
  data.term.c_oflag &= ~OLCUC;
  #endif
  #ifdef NLDLY
  data.term.c_oflag &= ~NLDLY;
  #endif
  #ifdef CRDLY
  data.term.c_oflag &= ~CRDLY;
  #endif
  #ifdef TABDLY
  data.term.c_oflag &= ~TABDLY;
  #endif
  #ifdef BSDLY
  data.term.c_oflag &= ~BSDLY;
  #endif
  #ifdef VTDLY
  data.term.c_oflag &= ~VTDLY;
  #endif
  #ifdef FFDLY
  data.term.c_oflag &= ~FFDLY;
  #endif
  #ifdef OFILL
  data.term.c_oflag &= ~OFILL;
  #endif
  data.term.c_lflag &= ~(ISIG | ICANON | ECHO | IEXTEN);
  data.term.c_cc[VMIN] = 0;
  data.term.c_cc[VTIME] = 0;

  /* Ignore modem status lines; enable receiver; leave control lines alone on close. */
  data.term.c_cflag |= (CLOCAL | CREAD);
  data.term.c_cflag &= ~(HUPCL);
#endif

#ifdef _WIN32
  if (ClearCommError(port->hdl, &errors, &status) == 0) return sp_fail_return(__func__, "ClearCommError() failed");
#endif

  ret = set_config(port, &data, &config);

  if (ret < 0) {
    sp_close(port);
    return sp_normalize_return(ret);
  }

  return SP_OK;
}

enum sp_return sp_close(struct sp_port *port) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  TLOG_DEBUG("Closing port {}", port->name);

#ifdef _WIN32
  /* Returns non-zero upon success, 0 upon failure. */
  if (CloseHandle(port->hdl) == 0) return sp_fail_return(__func__, "Port CloseHandle() failed");
  port->hdl = INVALID_HANDLE_VALUE;

  /* Close event handles for overlapped structures. */
  #define CLOSE_OVERLAPPED(ovl)                                                                    \
    do {                                                                                           \
      if (port->ovl.hEvent != INVALID_HANDLE_VALUE && CloseHandle(port->ovl.hEvent) == 0)          \
        return sp_fail_return(__func__, #ovl "event CloseHandle() failed");                        \
    } while (0)
  CLOSE_OVERLAPPED(read_ovl);
  CLOSE_OVERLAPPED(write_ovl);
  CLOSE_OVERLAPPED(wait_ovl);

  if (port->write_buf) {
    free(port->write_buf);
    port->write_buf = NULL;
  }
#else
  #ifdef TIOCNXCL
  ioctl(port->fd, TIOCNXCL);
  #endif

  /* Returns 0 upon success, -1 upon failure. */
  if (close(port->fd) == -1) return sp_fail_return(__func__, "close() failed");
  port->fd = -1;
#endif

  return SP_OK;
}

enum sp_return sp_flush(struct sp_port *port, enum sp_buffer buffers) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  if (buffers > SP_BUF_BOTH) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid buffer selection");

  const char *buffer_names[] = {"no", "input", "output", "both"};

  TLOG_DEBUG("Flushing {} buffers on port {}", buffer_names[buffers], port->name);

#ifdef _WIN32
  DWORD flags = 0;
  if (buffers & SP_BUF_INPUT) flags |= PURGE_RXCLEAR;
  if (buffers & SP_BUF_OUTPUT) flags |= PURGE_TXCLEAR;

  /* Returns non-zero upon success, 0 upon failure. */
  if (PurgeComm(port->hdl, flags) == 0) return sp_fail_return(__func__, "PurgeComm() failed");

  if (buffers & SP_BUF_INPUT) {
    enum sp_return ret = restart_wait(port);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }
#else
  int flags = 0;
  if (buffers == SP_BUF_BOTH) flags = TCIOFLUSH;
  else if (buffers == SP_BUF_INPUT) flags = TCIFLUSH;
  else if (buffers == SP_BUF_OUTPUT) flags = TCOFLUSH;

  /* Returns 0 upon success, -1 upon failure. */
  if (tcflush(port->fd, flags) < 0) return sp_fail_return(__func__, "tcflush() failed");
#endif
  return SP_OK;
}

enum sp_return sp_drain(struct sp_port *port) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  TLOG_DEBUG("Draining port {}", port->name);

#ifdef _WIN32
  /* Returns non-zero upon success, 0 upon failure. */
  if (FlushFileBuffers(port->hdl) == 0) return sp_fail_return(__func__, "FlushFileBuffers() failed");
  return SP_OK;
#else
  int result;
  while (1) {
  #if defined(__ANDROID__) && (__ANDROID_API__ < 21)
    /* Android only has tcdrain from platform 21 onwards.
     * On previous API versions, use the ioctl directly. */
    int arg = 1;
    result = ioctl(port->fd, TCSBRK, &arg);
  #else
    result = tcdrain(port->fd);
  #endif
    if (result < 0) {
      if (errno == EINTR) {
        TLOG_DEBUG("tcdrain() was interrupted");
        continue;
      } else {
        return sp_fail_return(__func__, "tcdrain() failed");
      }
    } else {
      return SP_OK;
    }
  }
#endif
}

#ifdef _WIN32
static enum sp_return await_write_completion(struct sp_port *port) {
  DWORD bytes_written;
  BOOL result;

  /* Wait for previous non-blocking write to complete, if any. */
  if (port->writing) {
    TLOG_DEBUG("Waiting for previous write to complete");
    result = GetOverlappedResult(port->hdl, &port->write_ovl, &bytes_written, TRUE);
    port->writing = 0;
    if (!result) return sp_fail_return(__func__, "Previous write failed to complete");
    TLOG_DEBUG("Previous write completed");
  }

  return SP_OK;
}
#endif

enum sp_return sp_blocking_write(struct sp_port *port, const void *buf, size_t count,
                                        unsigned int timeout_ms) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  if (!buf) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null buffer");

  if (count == 0) return 0;

#ifdef _WIN32
  DWORD remaining_ms, write_size, bytes_written;
  size_t remaining_bytes, total_bytes_written = 0;
  const uint8_t *write_ptr = (uint8_t *)buf;
  bool result;
  struct timeout timeout;

  timeout_start(&timeout, timeout_ms);

  {
    enum sp_return ret = await_write_completion(port);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }

  while (total_bytes_written < count) {

    if (timeout_check(&timeout)) break;

    remaining_ms = timeout_remaining_ms(&timeout);

    if (port->timeouts.WriteTotalTimeoutConstant != remaining_ms) {
      port->timeouts.WriteTotalTimeoutConstant = remaining_ms;
      if (SetCommTimeouts(port->hdl, &port->timeouts) == 0) return sp_fail_return(__func__, "SetCommTimeouts() failed");
    }

    /* Reduce write size if it exceeds the WriteFile limit. */
    remaining_bytes = count - total_bytes_written;
    if (remaining_bytes > WRITEFILE_MAX_SIZE) write_size = WRITEFILE_MAX_SIZE;
    else write_size = (DWORD)remaining_bytes;

    /* Start write. */

    result = WriteFile(port->hdl, write_ptr, write_size, NULL, &port->write_ovl);

    timeout_update(&timeout);

    if (result) {
      bytes_written = write_size;
    } else if (GetLastError() == ERROR_IO_PENDING) {
      if (GetOverlappedResult(port->hdl, &port->write_ovl, &bytes_written, TRUE) == 0) {
        if (GetLastError() == ERROR_SEM_TIMEOUT) {
          break;
        } else {
          return sp_fail_return(__func__, "GetOverlappedResult() failed");
        }
      }
    } else {
      return sp_fail_return(__func__, "WriteFile() failed");
    }

    write_ptr += bytes_written;
    total_bytes_written += bytes_written;
  }

  return (int)total_bytes_written;
#else
  size_t bytes_written = 0;
  unsigned char *ptr = (unsigned char *)buf;
  struct timeout timeout;
  fd_set fds;
  ssize_t result;

  timeout_start(&timeout, timeout_ms);

  FD_ZERO(&fds);
  FD_SET(port->fd, &fds);

  /* Loop until we have written the requested number of bytes. */
  while (bytes_written < count) {

    if (timeout_check(&timeout)) break;

    result = select(port->fd + 1, NULL, &fds, NULL, timeout_timeval(&timeout));

    timeout_update(&timeout);

    if (result < 0) {
      if (errno == EINTR) {
        continue;
      } else {
        return sp_fail_return(__func__, "select() failed");
      }
    } else if (result == 0) {
      /* Timeout has expired. */
      break;
    }

    /* Do write. */
    result = write(port->fd, ptr, count - bytes_written);

    if (result < 0) {
      if (errno == EAGAIN)
        /* This shouldn't happen because we did a select() first, but handle anyway. */
        continue;
      else /* This is an actual failure. */
        return sp_fail_return(__func__, "write() failed");
    }

    bytes_written += result;
    ptr += result;
  }

  return bytes_written;
#endif
}

enum sp_return sp_nonblocking_write(struct sp_port *port, const void *buf, size_t count) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  if (!buf) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null buffer");

  if (count == 0) return 0;

#ifdef _WIN32
  size_t buf_bytes;

  /* Check whether previous write is complete. */
  if (port->writing) {
    if (HasOverlappedIoCompleted(&port->write_ovl)) {
      port->writing = 0;
    } else {
      /* Can't take a new write until the previous one finishes. */
      return 0;
    }
  }

  /* Set timeout. */
  if (port->timeouts.WriteTotalTimeoutConstant != 0) {
    port->timeouts.WriteTotalTimeoutConstant = 0;
    if (SetCommTimeouts(port->hdl, &port->timeouts) == 0) return sp_fail_return(__func__, "SetCommTimeouts() failed");
  }

  /* Reduce count if it exceeds the WriteFile limit. */
  if (count > WRITEFILE_MAX_SIZE) count = WRITEFILE_MAX_SIZE;

  /* Copy data to our write buffer. */
  buf_bytes = min(port->write_buf_size, count);
  memcpy(port->write_buf, buf, buf_bytes);

  /* Start asynchronous write. */
  if (WriteFile(port->hdl, port->write_buf, (DWORD)buf_bytes, NULL, &port->write_ovl) == 0) {
    if (GetLastError() == ERROR_IO_PENDING) {
      port->writing = !HasOverlappedIoCompleted(&port->write_ovl);
    } else {
      /* Actual failure of some kind. */
      return sp_fail_return(__func__, "WriteFile() failed");
    }
  }

  return (int)buf_bytes;
#else
  /* Returns the number of bytes written, or -1 upon failure. */
  ssize_t written = write(port->fd, buf, count);

  if (written < 0) {
    if (errno == EAGAIN)
      // Buffer is full, no bytes written.
      return 0;
    else return sp_fail_return(__func__, "write() failed");
  } else {
    return written;
  }
#endif
}

#ifdef _WIN32
/* Restart wait operation if buffer was emptied. */
static enum sp_return restart_wait_if_needed(struct sp_port *port, unsigned int bytes_read) {
  DWORD errors;
  COMSTAT comstat;

  /* Only skip restarting the wait operation if we didn't have a
   * wakeup immediately following the exit of the last thread that
   * re-initiated the wait loop.
   */
  if (!port->last_wait_thread_exited && bytes_read == 0) return SP_OK;

  if (ClearCommError(port->hdl, &errors, &comstat) == 0) return sp_fail_return(__func__, "ClearCommError() failed");

  if (comstat.cbInQue == 0) {
    enum sp_return ret = restart_wait(port);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }

  return SP_OK;
}
#endif

enum sp_return sp_blocking_read(struct sp_port *port, void *buf, size_t count,
                                       unsigned int timeout_ms) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  if (!buf) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null buffer");

  if (count == 0) return 0;

#ifdef _WIN32
  DWORD bytes_read;

  /* Set timeout. */
  if (port->timeouts.ReadIntervalTimeout != 0 || port->timeouts.ReadTotalTimeoutMultiplier != 0 ||
      port->timeouts.ReadTotalTimeoutConstant != timeout_ms) {
    port->timeouts.ReadIntervalTimeout = 0;
    port->timeouts.ReadTotalTimeoutMultiplier = 0;
    port->timeouts.ReadTotalTimeoutConstant = timeout_ms;
    if (SetCommTimeouts(port->hdl, &port->timeouts) == 0) return sp_fail_return(__func__, "SetCommTimeouts() failed");
  }

  /* Start read. */
  if (ReadFile(port->hdl, buf, (DWORD)count, NULL, &port->read_ovl)) {
    bytes_read = (DWORD)count;
  } else if (GetLastError() == ERROR_IO_PENDING) {
    if (GetOverlappedResult(port->hdl, &port->read_ovl, &bytes_read, TRUE) == 0)
      return sp_fail_return(__func__, "GetOverlappedResult() failed");
  } else {
    return sp_fail_return(__func__, "ReadFile() failed");
  }

  {
    enum sp_return ret = restart_wait_if_needed(port, bytes_read);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }

  return (int)bytes_read;

#else
  size_t bytes_read = 0;
  unsigned char *ptr = (unsigned char *)buf;
  struct timeout timeout;
  fd_set fds;
  ssize_t result;

  timeout_start(&timeout, timeout_ms);

  FD_ZERO(&fds);
  FD_SET(port->fd, &fds);

  /* Loop until we have the requested number of bytes. */
  while (bytes_read < count) {

    if (timeout_check(&timeout)) /* Timeout has expired. */
      break;

    result = select(port->fd + 1, &fds, NULL, NULL, timeout_timeval(&timeout));

    timeout_update(&timeout);

    if (result < 0) {
      if (errno == EINTR) {
        continue;
      } else {
        return sp_fail_return(__func__, "select() failed");
      }
    } else if (result == 0) {
      /* Timeout has expired. */
      break;
    }

    /* Do read. */
    result = read(port->fd, ptr, count - bytes_read);

    if (result < 0) {
      if (errno == EAGAIN)
        /*
         * This shouldn't happen because we did a
         * select() first, but handle anyway.
         */
        continue;
      else /* This is an actual failure. */
        return sp_fail_return(__func__, "read() failed");
    }

    bytes_read += result;
    ptr += result;
  }

  return bytes_read;
#endif
}

enum sp_return sp_blocking_read_next(struct sp_port *port, void *buf, size_t count,
                                            unsigned int timeout_ms) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  if (!buf) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null buffer");

  if (count == 0) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Zero count");

#ifdef _WIN32
  DWORD bytes_read = 0;

  /* If timeout_ms == 0, set maximum timeout. */
  DWORD timeout_val = (timeout_ms == 0 ? MAXDWORD - 1 : timeout_ms);

  /* Set timeout. */
  if (port->timeouts.ReadIntervalTimeout != MAXDWORD ||
      port->timeouts.ReadTotalTimeoutMultiplier != MAXDWORD ||
      port->timeouts.ReadTotalTimeoutConstant != timeout_val) {
    port->timeouts.ReadIntervalTimeout = MAXDWORD;
    port->timeouts.ReadTotalTimeoutMultiplier = MAXDWORD;
    port->timeouts.ReadTotalTimeoutConstant = timeout_val;
    if (SetCommTimeouts(port->hdl, &port->timeouts) == 0) return sp_fail_return(__func__, "SetCommTimeouts() failed");
  }

  /* Loop until we have at least one byte, or timeout is reached. */
  while (bytes_read == 0) {
    /* Start read. */
    if (!ReadFile(port->hdl, buf, (DWORD)count, &bytes_read, &port->read_ovl)) {
      if (GetLastError() == ERROR_IO_PENDING) {
        if (GetOverlappedResult(port->hdl, &port->read_ovl, &bytes_read, TRUE) == 0)
          return sp_fail_return(__func__, "GetOverlappedResult() failed");
        if (bytes_read == 0 && timeout_ms > 0) break;
      } else {
        return sp_fail_return(__func__, "ReadFile() failed");
      }
    }
  }

  {
    enum sp_return ret = restart_wait_if_needed(port, bytes_read);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }

  return bytes_read;

#else
  size_t bytes_read = 0;
  struct timeout timeout;
  fd_set fds;
  ssize_t result;

  timeout_start(&timeout, timeout_ms);

  FD_ZERO(&fds);
  FD_SET(port->fd, &fds);

  /* Loop until we have at least one byte, or timeout is reached. */
  while (bytes_read == 0) {

    if (timeout_check(&timeout)) /* Timeout has expired. */
      break;

    result = select(port->fd + 1, &fds, NULL, NULL, timeout_timeval(&timeout));

    timeout_update(&timeout);

    if (result < 0) {
      if (errno == EINTR) {
        continue;
      } else {
        return sp_fail_return(__func__, "select() failed");
      }
    } else if (result == 0) {
      /* Timeout has expired. */
      break;
    }

    /* Do read. */
    result = read(port->fd, buf, count);

    if (result < 0) {
      if (errno == EAGAIN)
        /* This shouldn't happen because we did a select() first, but handle anyway. */
        continue;
      else /* This is an actual failure. */
        return sp_fail_return(__func__, "read() failed");
    }

    bytes_read = result;
  }

  return bytes_read;
#endif
}

enum sp_return sp_nonblocking_read(struct sp_port *port, void *buf, size_t count) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  if (!buf) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null buffer");

#ifdef _WIN32
  DWORD bytes_read;

  /* Set timeout. */
  if (port->timeouts.ReadIntervalTimeout != MAXDWORD ||
      port->timeouts.ReadTotalTimeoutMultiplier != 0 ||
      port->timeouts.ReadTotalTimeoutConstant != 0) {
    port->timeouts.ReadIntervalTimeout = MAXDWORD;
    port->timeouts.ReadTotalTimeoutMultiplier = 0;
    port->timeouts.ReadTotalTimeoutConstant = 0;
    if (SetCommTimeouts(port->hdl, &port->timeouts) == 0) return sp_fail_return(__func__, "SetCommTimeouts() failed");
  }

  /* Do read. */
  if (ReadFile(port->hdl, buf, (DWORD)count, NULL, &port->read_ovl) == 0)
    if (GetLastError() != ERROR_IO_PENDING) return sp_fail_return(__func__, "ReadFile() failed");

  /* Get number of bytes read. */
  if (GetOverlappedResult(port->hdl, &port->read_ovl, &bytes_read, FALSE) == 0)
    return sp_fail_return(__func__, "GetOverlappedResult() failed");

  {
    enum sp_return ret = restart_wait_if_needed(port, bytes_read);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }

  return bytes_read;
#else
  ssize_t bytes_read;

  /* Returns the number of bytes read, or -1 upon failure. */
  if ((bytes_read = read(port->fd, buf, count)) < 0) {
    if (errno == EAGAIN) /* No bytes available. */
      bytes_read = 0;
    else /* This is an actual failure. */
      return sp_fail_return(__func__, "read() failed");
  }
  return bytes_read;
#endif
}

enum sp_return sp_input_waiting(struct sp_port *port) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  TLOG_DEBUG("Checking input bytes waiting on port {}", port->name);

#ifdef _WIN32
  DWORD errors;
  COMSTAT comstat;

  if (ClearCommError(port->hdl, &errors, &comstat) == 0) return sp_fail_return(__func__, "ClearCommError() failed");
  return comstat.cbInQue;
#else
  int bytes_waiting;
  if (ioctl(port->fd, TIOCINQ, &bytes_waiting) < 0) return sp_fail_return(__func__, "TIOCINQ ioctl failed");
  return bytes_waiting;
#endif
}

enum sp_return sp_output_waiting(struct sp_port *port) {
#ifdef __CYGWIN__
  /* TIOCOUTQ is not defined in Cygwin headers */
  return sp_error_return(__func__, SP_ERR_SUPP, "SP_ERR_SUPP", "Getting output bytes waiting is not supported on Cygwin");
#else
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  TLOG_DEBUG("Checking output bytes waiting on port {}", port->name);

  #ifdef _WIN32
  DWORD errors;
  COMSTAT comstat;

  if (ClearCommError(port->hdl, &errors, &comstat) == 0) return sp_fail_return(__func__, "ClearCommError() failed");
  return comstat.cbOutQue;
  #else
  int bytes_waiting;
  if (ioctl(port->fd, TIOCOUTQ, &bytes_waiting) < 0) return sp_fail_return(__func__, "TIOCOUTQ ioctl failed");
  return bytes_waiting;
  #endif
#endif
}

enum sp_return sp_new_event_set(struct sp_event_set **result_ptr) {
  struct sp_event_set *result;
  if (!result_ptr) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null result");

  *result_ptr = NULL;

  if (!(result = malloc(sizeof(struct sp_event_set))))
    return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "sp_event_set malloc() failed");

  memset(result, 0, sizeof(struct sp_event_set));

  *result_ptr = result;

  return SP_OK;
}

static enum sp_return add_handle(struct sp_event_set *event_set, event_handle handle,
                                 enum sp_event mask) {
  void *new_handles;
  enum sp_event *new_masks;
  if (!(new_handles = realloc(event_set->handles, sizeof(event_handle) * (event_set->count + 1))))
    return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "Handle array realloc() failed");

  event_set->handles = new_handles;

  if (!(new_masks = realloc(event_set->masks, sizeof(enum sp_event) * (event_set->count + 1))))
    return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "Mask array realloc() failed");

  event_set->masks = new_masks;

  ((event_handle *)event_set->handles)[event_set->count] = handle;
  event_set->masks[event_set->count] = mask;

  event_set->count++;

  return SP_OK;
}

enum sp_return sp_add_port_events(struct sp_event_set *event_set, const struct sp_port *port,
                                         enum sp_event mask) {
  if (!event_set) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null event set");

  if (!port) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null port");

  if (mask > (SP_EVENT_RX_READY | SP_EVENT_TX_READY | SP_EVENT_ERROR))
    return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid event mask");

  if (!mask) return SP_OK;

#ifdef _WIN32
  enum sp_event handle_mask;
  if ((handle_mask = mask & SP_EVENT_TX_READY))
    {
      enum sp_return ret = add_handle(event_set, port->write_ovl.hEvent, handle_mask);
      if (ret != SP_OK) return sp_normalize_return(ret);
    }
  if ((handle_mask = mask & (SP_EVENT_RX_READY | SP_EVENT_ERROR)))
    {
      enum sp_return ret = add_handle(event_set, port->wait_ovl.hEvent, handle_mask);
      if (ret != SP_OK) return sp_normalize_return(ret);
    }
#else
  {
    enum sp_return ret = add_handle(event_set, port->fd, mask);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }
#endif

  return SP_OK;
}

void sp_free_event_set(struct sp_event_set *event_set) {
  if (!event_set) {
    TLOG_DEBUG("Null event set");
    return;
  }

  TLOG_DEBUG("Freeing event set");

  if (event_set->handles) free(event_set->handles);
  if (event_set->masks) free(event_set->masks);

  free(event_set);

  return;
}

enum sp_return sp_wait(struct sp_event_set *event_set, unsigned int timeout_ms) {
  if (!event_set) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null event set");

#ifdef _WIN32
  if (WaitForMultipleObjects(event_set->count, event_set->handles, FALSE,
                             timeout_ms ? timeout_ms : INFINITE) == WAIT_FAILED)
    return sp_fail_return(__func__, "WaitForMultipleObjects() failed");

  return SP_OK;
#else
  struct timeout timeout;
  int poll_timeout;
  int result;
  struct pollfd *pollfds;
  unsigned int i;

  if (!(pollfds = malloc(sizeof(struct pollfd) * event_set->count)))
    return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "pollfds malloc() failed");

  for (i = 0; i < event_set->count; i++) {
    pollfds[i].fd = ((int *)event_set->handles)[i];
    pollfds[i].events = 0;
    pollfds[i].revents = 0;
    if (event_set->masks[i] & SP_EVENT_RX_READY) pollfds[i].events |= POLLIN;
    if (event_set->masks[i] & SP_EVENT_TX_READY) pollfds[i].events |= POLLOUT;
    if (event_set->masks[i] & SP_EVENT_ERROR) pollfds[i].events |= POLLERR;
  }

  timeout_start(&timeout, timeout_ms);
  timeout_limit(&timeout, INT_MAX);

  /* Loop until an event occurs. */
  while (1) {

    if (timeout_check(&timeout)) {
      TLOG_DEBUG("Wait timed out");
      break;
    }

    poll_timeout = (int)timeout_remaining_ms(&timeout);
    if (poll_timeout == 0) poll_timeout = -1;

    result = poll(pollfds, event_set->count, poll_timeout);

    timeout_update(&timeout);

    if (result < 0) {
      if (errno == EINTR) {
        TLOG_DEBUG("poll() call was interrupted, repeating");
        continue;
      } else {
        free(pollfds);
        return sp_fail_return(__func__, "poll() failed");
      }
    } else if (result == 0) {
      TLOG_DEBUG("poll() timed out");
      if (!timeout.overflow) break;
    } else {
      TLOG_DEBUG("poll() completed");
      break;
    }
  }

  free(pollfds);
  return SP_OK;
#endif
}

#ifdef USE_TERMIOS_SPEED
static enum sp_return get_baudrate(int fd, int *baudrate) {
  void *data;
  TLOG_DEBUG("Getting baud rate");

  if (!(data = malloc(get_termios_size()))) return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "termios malloc failed");

  if (ioctl(fd, get_termios_get_ioctl(), data) < 0) {
    free(data);
    return sp_fail_return(__func__, "Getting termios failed");
  }

  *baudrate = get_termios_speed(data);

  free(data);

  return SP_OK;
}

static enum sp_return set_baudrate(int fd, int baudrate) {
  void *data;
  TLOG_DEBUG("Getting baud rate");

  if (!(data = malloc(get_termios_size()))) return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "termios malloc failed");

  if (ioctl(fd, get_termios_get_ioctl(), data) < 0) {
    free(data);
    return sp_fail_return(__func__, "Getting termios failed");
  }

  TLOG_DEBUG("Setting baud rate");

  set_termios_speed(data, baudrate);

  if (ioctl(fd, get_termios_set_ioctl(), data) < 0) {
    free(data);
    return sp_fail_return(__func__, "Setting termios failed");
  }

  free(data);

  return SP_OK;
}
#endif /* USE_TERMIOS_SPEED */

#ifdef USE_TERMIOX
static enum sp_return get_flow(int fd, struct port_data *data) {
  void *termx;
  TLOG_DEBUG("Getting advanced flow control");

  if (!(termx = malloc(get_termiox_size()))) return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "termiox malloc failed");

  if (ioctl(fd, TCGETX, termx) < 0) {
    free(termx);
    return sp_fail_return(__func__, "Getting termiox failed");
  }

  get_termiox_flow(termx, &data->rts_flow, &data->cts_flow, &data->dtr_flow, &data->dsr_flow);

  free(termx);

  return SP_OK;
}

static enum sp_return set_flow(int fd, struct port_data *data) {
  void *termx;
  TLOG_DEBUG("Getting advanced flow control");

  if (!(termx = malloc(get_termiox_size()))) return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "termiox malloc failed");

  if (ioctl(fd, TCGETX, termx) < 0) {
    free(termx);
    return sp_fail_return(__func__, "Getting termiox failed");
  }

  TLOG_DEBUG("Setting advanced flow control");

  set_termiox_flow(termx, data->rts_flow, data->cts_flow, data->dtr_flow, data->dsr_flow);

  if (ioctl(fd, TCSETX, termx) < 0) {
    free(termx);
    return sp_fail_return(__func__, "Setting termiox failed");
  }

  free(termx);

  return SP_OK;
}
#endif /* USE_TERMIOX */

static enum sp_return get_config(struct sp_port *port, struct port_data *data,
                                 struct sp_port_config *config) {
  unsigned int i;
  TLOG_DEBUG("Getting configuration for port {}", port->name);

#ifdef _WIN32
  if (!GetCommState(port->hdl, &data->dcb)) return sp_fail_return(__func__, "GetCommState() failed");

  for (i = 0; i < NUM_STD_BAUDRATES; i++) {
    if (data->dcb.BaudRate == std_baudrates[i].index) {
      config->baudrate = std_baudrates[i].value;
      break;
    }
  }

  if (i == NUM_STD_BAUDRATES) /* BaudRate field can be either an index or a custom baud rate. */
    config->baudrate = data->dcb.BaudRate;

  config->bits = data->dcb.ByteSize;

  switch (data->dcb.Parity) {
  case NOPARITY:
    config->parity = SP_PARITY_NONE;
    break;
  case ODDPARITY:
    config->parity = SP_PARITY_ODD;
    break;
  case EVENPARITY:
    config->parity = SP_PARITY_EVEN;
    break;
  case MARKPARITY:
    config->parity = SP_PARITY_MARK;
    break;
  case SPACEPARITY:
    config->parity = SP_PARITY_SPACE;
    break;
  default:
    config->parity = -1;
  }

  switch (data->dcb.StopBits) {
  case ONESTOPBIT:
    config->stopbits = 1;
    break;
  case TWOSTOPBITS:
    config->stopbits = 2;
    break;
  default:
    config->stopbits = -1;
  }

  switch (data->dcb.fRtsControl) {
  case RTS_CONTROL_DISABLE:
    config->rts = SP_RTS_OFF;
    break;
  case RTS_CONTROL_ENABLE:
    config->rts = SP_RTS_ON;
    break;
  case RTS_CONTROL_HANDSHAKE:
    config->rts = SP_RTS_FLOW_CONTROL;
    break;
  default:
    config->rts = -1;
  }

  config->cts = data->dcb.fOutxCtsFlow ? SP_CTS_FLOW_CONTROL : SP_CTS_IGNORE;

  switch (data->dcb.fDtrControl) {
  case DTR_CONTROL_DISABLE:
    config->dtr = SP_DTR_OFF;
    break;
  case DTR_CONTROL_ENABLE:
    config->dtr = SP_DTR_ON;
    break;
  case DTR_CONTROL_HANDSHAKE:
    config->dtr = SP_DTR_FLOW_CONTROL;
    break;
  default:
    config->dtr = -1;
  }

  config->dsr = data->dcb.fOutxDsrFlow ? SP_DSR_FLOW_CONTROL : SP_DSR_IGNORE;

  if (data->dcb.fInX) {
    if (data->dcb.fOutX) config->xon_xoff = SP_XONXOFF_INOUT;
    else config->xon_xoff = SP_XONXOFF_IN;
  } else {
    if (data->dcb.fOutX) config->xon_xoff = SP_XONXOFF_OUT;
    else config->xon_xoff = SP_XONXOFF_DISABLED;
  }

#else // !_WIN32

  if (tcgetattr(port->fd, &data->term) < 0) return sp_fail_return(__func__, "tcgetattr() failed");

  if (ioctl(port->fd, TIOCMGET, &data->controlbits) < 0) return sp_fail_return(__func__, "TIOCMGET ioctl failed");

  #ifdef USE_TERMIOX
  int ret = get_flow(port->fd, data);

  if (ret == SP_ERR_FAIL && errno == EINVAL) data->termiox_supported = 0;
  else if (ret < 0) return sp_normalize_return(ret);
  else data->termiox_supported = 1;
  #else
  data->termiox_supported = 0;
  #endif

  for (i = 0; i < NUM_STD_BAUDRATES; i++) {
    if (cfgetispeed(&data->term) == std_baudrates[i].index) {
      config->baudrate = std_baudrates[i].value;
      break;
    }
  }

  if (i == NUM_STD_BAUDRATES) {
  #ifdef __APPLE__
    config->baudrate = (int)data->term.c_ispeed;
  #elif defined(USE_TERMIOS_SPEED)
    {
      enum sp_return ret = get_baudrate(port->fd, &config->baudrate);
      if (ret != SP_OK) return sp_normalize_return(ret);
    }
  #else
    config->baudrate = -1;
  #endif
  }

  switch (data->term.c_cflag & CSIZE) {
  case CS8:
    config->bits = 8;
    break;
  case CS7:
    config->bits = 7;
    break;
  case CS6:
    config->bits = 6;
    break;
  case CS5:
    config->bits = 5;
    break;
  default:
    config->bits = -1;
  }

  if (!(data->term.c_cflag & PARENB) && (data->term.c_iflag & IGNPAR))
    config->parity = SP_PARITY_NONE;
  else if (!(data->term.c_cflag & PARENB) || (data->term.c_iflag & IGNPAR)) config->parity = -1;
  #ifdef CMSPAR
  else if (data->term.c_cflag & CMSPAR)
    config->parity = (data->term.c_cflag & PARODD) ? SP_PARITY_MARK : SP_PARITY_SPACE;
  #endif
  else config->parity = (data->term.c_cflag & PARODD) ? SP_PARITY_ODD : SP_PARITY_EVEN;

  config->stopbits = (data->term.c_cflag & CSTOPB) ? 2 : 1;

  if (data->term.c_cflag & CRTSCTS) {
    config->rts = SP_RTS_FLOW_CONTROL;
    config->cts = SP_CTS_FLOW_CONTROL;
  } else {
    if (data->termiox_supported && data->rts_flow) config->rts = SP_RTS_FLOW_CONTROL;
    else config->rts = (data->controlbits & TIOCM_RTS) ? SP_RTS_ON : SP_RTS_OFF;

    config->cts = (data->termiox_supported && data->cts_flow) ? SP_CTS_FLOW_CONTROL : SP_CTS_IGNORE;
  }

  if (data->termiox_supported && data->dtr_flow) config->dtr = SP_DTR_FLOW_CONTROL;
  else config->dtr = (data->controlbits & TIOCM_DTR) ? SP_DTR_ON : SP_DTR_OFF;

  config->dsr = (data->termiox_supported && data->dsr_flow) ? SP_DSR_FLOW_CONTROL : SP_DSR_IGNORE;

  if (data->term.c_iflag & IXOFF) {
    if (data->term.c_iflag & IXON) config->xon_xoff = SP_XONXOFF_INOUT;
    else config->xon_xoff = SP_XONXOFF_IN;
  } else {
    if (data->term.c_iflag & IXON) config->xon_xoff = SP_XONXOFF_OUT;
    else config->xon_xoff = SP_XONXOFF_DISABLED;
  }
#endif

  return SP_OK;
}

static enum sp_return set_config(struct sp_port *port, struct port_data *data,
                                 const struct sp_port_config *config) {
  unsigned int i;
#ifdef __APPLE__
  BAUD_TYPE baud_nonstd;

  baud_nonstd = B0;
#endif
#ifdef USE_TERMIOS_SPEED
  int baud_nonstd = 0;
#endif
  TLOG_DEBUG("Setting configuration for port {}", port->name);

#ifdef _WIN32
  BYTE *new_buf;

  {
    enum sp_return ret = await_write_completion(port);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }

  if (config->baudrate >= 0) {
    for (i = 0; i < NUM_STD_BAUDRATES; i++) {
      if (config->baudrate == std_baudrates[i].value) {
        data->dcb.BaudRate = std_baudrates[i].index;
        break;
      }
    }

    if (i == NUM_STD_BAUDRATES) data->dcb.BaudRate = config->baudrate;

    /* Allocate write buffer for 50ms of data at baud rate. */
    port->write_buf_size = max(config->baudrate / (8 * 20), 1);
    new_buf = realloc(port->write_buf, port->write_buf_size);
    if (!new_buf) return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "Allocating write buffer failed");
    port->write_buf = new_buf;
  }

  if (config->bits >= 0) data->dcb.ByteSize = config->bits;

  if (config->parity >= 0) {
    switch (config->parity) {
    case SP_PARITY_NONE:
      data->dcb.Parity = NOPARITY;
      break;
    case SP_PARITY_ODD:
      data->dcb.Parity = ODDPARITY;
      break;
    case SP_PARITY_EVEN:
      data->dcb.Parity = EVENPARITY;
      break;
    case SP_PARITY_MARK:
      data->dcb.Parity = MARKPARITY;
      break;
    case SP_PARITY_SPACE:
      data->dcb.Parity = SPACEPARITY;
      break;
    default:
      return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid parity setting");
    }
  }

  if (config->stopbits >= 0) {
    switch (config->stopbits) {
    /* Note: There's also ONE5STOPBITS == 1.5 (unneeded so far). */
    case 1:
      data->dcb.StopBits = ONESTOPBIT;
      break;
    case 2:
      data->dcb.StopBits = TWOSTOPBITS;
      break;
    default:
      return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid stop bit setting");
    }
  }

  if (config->rts >= 0) {
    switch (config->rts) {
    case SP_RTS_OFF:
      data->dcb.fRtsControl = RTS_CONTROL_DISABLE;
      break;
    case SP_RTS_ON:
      data->dcb.fRtsControl = RTS_CONTROL_ENABLE;
      break;
    case SP_RTS_FLOW_CONTROL:
      data->dcb.fRtsControl = RTS_CONTROL_HANDSHAKE;
      break;
    default:
      return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid RTS setting");
    }
  }

  if (config->cts >= 0) {
    switch (config->cts) {
    case SP_CTS_IGNORE:
      data->dcb.fOutxCtsFlow = FALSE;
      break;
    case SP_CTS_FLOW_CONTROL:
      data->dcb.fOutxCtsFlow = TRUE;
      break;
    default:
      return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid CTS setting");
    }
  }

  if (config->dtr >= 0) {
    switch (config->dtr) {
    case SP_DTR_OFF:
      data->dcb.fDtrControl = DTR_CONTROL_DISABLE;
      break;
    case SP_DTR_ON:
      data->dcb.fDtrControl = DTR_CONTROL_ENABLE;
      break;
    case SP_DTR_FLOW_CONTROL:
      data->dcb.fDtrControl = DTR_CONTROL_HANDSHAKE;
      break;
    default:
      return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid DTR setting");
    }
  }

  if (config->dsr >= 0) {
    switch (config->dsr) {
    case SP_DSR_IGNORE:
      data->dcb.fOutxDsrFlow = FALSE;
      break;
    case SP_DSR_FLOW_CONTROL:
      data->dcb.fOutxDsrFlow = TRUE;
      break;
    default:
      return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid DSR setting");
    }
  }

  if (config->xon_xoff >= 0) {
    switch (config->xon_xoff) {
    case SP_XONXOFF_DISABLED:
      data->dcb.fInX = FALSE;
      data->dcb.fOutX = FALSE;
      break;
    case SP_XONXOFF_IN:
      data->dcb.fInX = TRUE;
      data->dcb.fOutX = FALSE;
      break;
    case SP_XONXOFF_OUT:
      data->dcb.fInX = FALSE;
      data->dcb.fOutX = TRUE;
      break;
    case SP_XONXOFF_INOUT:
      data->dcb.fInX = TRUE;
      data->dcb.fOutX = TRUE;
      break;
    default:
      return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid XON/XOFF setting");
    }
  }

  if (!SetCommState(port->hdl, &data->dcb)) return sp_fail_return(__func__, "SetCommState() failed");

#else /* !_WIN32 */

  int controlbits;

  if (config->baudrate >= 0) {
    for (i = 0; i < NUM_STD_BAUDRATES; i++) {
      if (config->baudrate == std_baudrates[i].value) {
        if (cfsetospeed(&data->term, std_baudrates[i].index) < 0)
          return sp_fail_return(__func__, "cfsetospeed() failed");

        if (cfsetispeed(&data->term, std_baudrates[i].index) < 0)
          return sp_fail_return(__func__, "cfsetispeed() failed");
        break;
      }
    }

    /* Non-standard baud rate */
    if (i == NUM_STD_BAUDRATES) {
  #ifdef __APPLE__
      /* Set "dummy" baud rate. */
      if (cfsetspeed(&data->term, B9600) < 0) return sp_fail_return(__func__, "cfsetspeed() failed");
      baud_nonstd = config->baudrate;
  #elif defined(USE_TERMIOS_SPEED)
      baud_nonstd = 1;
  #else
      return sp_error_return(__func__, SP_ERR_SUPP, "SP_ERR_SUPP", "Non-standard baudrate not supported");
  #endif
    }
  }

  if (config->bits >= 0) {
    data->term.c_cflag &= ~CSIZE;
    switch (config->bits) {
    case 8:
      data->term.c_cflag |= CS8;
      break;
    case 7:
      data->term.c_cflag |= CS7;
      break;
    case 6:
      data->term.c_cflag |= CS6;
      break;
    case 5:
      data->term.c_cflag |= CS5;
      break;
    default:
      return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid data bits setting");
    }
  }

  if (config->parity >= 0) {
    data->term.c_iflag &= ~IGNPAR;
    data->term.c_cflag &= ~(PARENB | PARODD);
  #ifdef CMSPAR
    data->term.c_cflag &= ~CMSPAR;
  #endif
    switch (config->parity) {
    case SP_PARITY_NONE:
      data->term.c_iflag |= IGNPAR;
      break;
    case SP_PARITY_EVEN:
      data->term.c_cflag |= PARENB;
      break;
    case SP_PARITY_ODD:
      data->term.c_cflag |= PARENB | PARODD;
      break;
  #ifdef CMSPAR
    case SP_PARITY_MARK:
      data->term.c_cflag |= PARENB | PARODD;
      data->term.c_cflag |= CMSPAR;
      break;
    case SP_PARITY_SPACE:
      data->term.c_cflag |= PARENB;
      data->term.c_cflag |= CMSPAR;
      break;
  #else
    case SP_PARITY_MARK:
    case SP_PARITY_SPACE:
      return sp_error_return(__func__, SP_ERR_SUPP, "SP_ERR_SUPP", "Mark/space parity not supported");
  #endif
    default:
      return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid parity setting");
    }
  }

  if (config->stopbits >= 0) {
    data->term.c_cflag &= ~CSTOPB;
    switch (config->stopbits) {
    case 1:
      data->term.c_cflag &= ~CSTOPB;
      break;
    case 2:
      data->term.c_cflag |= CSTOPB;
      break;
    default:
      return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid stop bits setting");
    }
  }

  if (config->rts >= 0 || config->cts >= 0) {
    if (data->termiox_supported) {
      data->rts_flow = data->cts_flow = 0;
      switch (config->rts) {
      case SP_RTS_OFF:
      case SP_RTS_ON:
        controlbits = TIOCM_RTS;
        if (ioctl(port->fd, config->rts == SP_RTS_ON ? TIOCMBIS : TIOCMBIC, &controlbits) < 0)
          return sp_fail_return(__func__, "Setting RTS signal level failed");
        break;
      case SP_RTS_FLOW_CONTROL:
        data->rts_flow = 1;
        break;
      default:
        break;
      }
      if (config->cts == SP_CTS_FLOW_CONTROL) data->cts_flow = 1;

      if (data->rts_flow && data->cts_flow) data->term.c_iflag |= CRTSCTS;
      else data->term.c_iflag &= ~CRTSCTS;
    } else {
      /* Asymmetric use of RTS/CTS not supported. */
      if (data->term.c_iflag & CRTSCTS) {
        /* Flow control can only be disabled for both RTS & CTS together. */
        if (config->rts >= 0 && config->rts != SP_RTS_FLOW_CONTROL) {
          if (config->cts != SP_CTS_IGNORE)
            return sp_error_return(__func__, SP_ERR_SUPP, "SP_ERR_SUPP", "RTS & CTS flow control must be disabled together");
        }
        if (config->cts >= 0 && config->cts != SP_CTS_FLOW_CONTROL) {
          if (config->rts <= 0 || config->rts == SP_RTS_FLOW_CONTROL)
            return sp_error_return(__func__, SP_ERR_SUPP, "SP_ERR_SUPP", "RTS & CTS flow control must be disabled together");
        }
      } else {
        /* Flow control can only be enabled for both RTS & CTS together. */
        if (((config->rts == SP_RTS_FLOW_CONTROL) && (config->cts != SP_CTS_FLOW_CONTROL)) ||
            ((config->cts == SP_CTS_FLOW_CONTROL) && (config->rts != SP_RTS_FLOW_CONTROL)))
          return sp_error_return(__func__, SP_ERR_SUPP, "SP_ERR_SUPP", "RTS & CTS flow control must be enabled together");
      }

      if (config->rts >= 0) {
        if (config->rts == SP_RTS_FLOW_CONTROL) {
          data->term.c_iflag |= CRTSCTS;
        } else {
          controlbits = TIOCM_RTS;
          if (ioctl(port->fd, config->rts == SP_RTS_ON ? TIOCMBIS : TIOCMBIC, &controlbits) < 0)
            return sp_fail_return(__func__, "Setting RTS signal level failed");
        }
      }
    }
  }

  if (config->dtr >= 0 || config->dsr >= 0) {
    if (data->termiox_supported) {
      data->dtr_flow = data->dsr_flow = 0;
      switch (config->dtr) {
      case SP_DTR_OFF:
      case SP_DTR_ON:
        controlbits = TIOCM_DTR;
        if (ioctl(port->fd, config->dtr == SP_DTR_ON ? TIOCMBIS : TIOCMBIC, &controlbits) < 0)
          return sp_fail_return(__func__, "Setting DTR signal level failed");
        break;
      case SP_DTR_FLOW_CONTROL:
        data->dtr_flow = 1;
        break;
      default:
        break;
      }
      if (config->dsr == SP_DSR_FLOW_CONTROL) data->dsr_flow = 1;
    } else {
      /* DTR/DSR flow control not supported. */
      if (config->dtr == SP_DTR_FLOW_CONTROL || config->dsr == SP_DSR_FLOW_CONTROL)
        return sp_error_return(__func__, SP_ERR_SUPP, "SP_ERR_SUPP", "DTR/DSR flow control not supported");

      if (config->dtr >= 0) {
        controlbits = TIOCM_DTR;
        if (ioctl(port->fd, config->dtr == SP_DTR_ON ? TIOCMBIS : TIOCMBIC, &controlbits) < 0)
          return sp_fail_return(__func__, "Setting DTR signal level failed");
      }
    }
  }

  if (config->xon_xoff >= 0) {
    data->term.c_iflag &= ~(IXON | IXOFF | IXANY);
    switch (config->xon_xoff) {
    case SP_XONXOFF_DISABLED:
      break;
    case SP_XONXOFF_IN:
      data->term.c_iflag |= IXOFF;
      break;
    case SP_XONXOFF_OUT:
      data->term.c_iflag |= IXON | IXANY;
      break;
    case SP_XONXOFF_INOUT:
      data->term.c_iflag |= IXON | IXOFF | IXANY;
      break;
    default:
      return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid XON/XOFF setting");
    }
  }

  if (tcsetattr(port->fd, TCSANOW, &data->term) < 0) return sp_fail_return(__func__, "tcsetattr() failed");

  #ifdef __APPLE__
  if (baud_nonstd != B0) {
    if (ioctl(port->fd, IOSSIOSPEED, &baud_nonstd) == -1) return sp_fail_return(__func__, "IOSSIOSPEED ioctl failed");
    /*
     * Set baud rates in data->term to correct, but incompatible
     * with tcsetattr() value, same as delivered by tcgetattr().
     */
    if (cfsetspeed(&data->term, baud_nonstd) < 0) return sp_fail_return(__func__, "cfsetspeed() failed");
  }
  #elif defined(__linux__)
    #ifdef USE_TERMIOS_SPEED
  if (baud_nonstd) {
    enum sp_return ret = set_baudrate(port->fd, config->baudrate);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }
    #endif
    #ifdef USE_TERMIOX
  if (data->termiox_supported) {
    enum sp_return ret = set_flow(port->fd, data);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }
    #endif
  #endif

#endif /* !_WIN32 */

  return SP_OK;
}

enum sp_return sp_new_config(struct sp_port_config **config_ptr) {
  struct sp_port_config *config;
  if (!config_ptr) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null result pointer");

  *config_ptr = NULL;

  if (!(config = malloc(sizeof(struct sp_port_config))))
    return sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "Config malloc failed");

  config->baudrate = -1;
  config->bits = -1;
  config->parity = -1;
  config->stopbits = -1;
  config->rts = -1;
  config->cts = -1;
  config->dtr = -1;
  config->dsr = -1;
  config->xon_xoff = -1;

  *config_ptr = config;

  return SP_OK;
}

void sp_free_config(struct sp_port_config *config) {
  if (!config) TLOG_DEBUG("Null config");
  else free(config);

  return;
}

enum sp_return sp_get_config(struct sp_port *port, struct sp_port_config *config) {
  struct port_data data;
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  if (!config) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null config");

  {
    enum sp_return ret = get_config(port, &data, config);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }

  return SP_OK;
}

enum sp_return sp_set_config(struct sp_port *port, const struct sp_port_config *config) {
  struct port_data data;
  struct sp_port_config prev_config;
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  if (!config) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null config");

  {
    enum sp_return ret = get_config(port, &data, &prev_config);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }
  {
    enum sp_return ret = set_config(port, &data, config);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }

  return SP_OK;
}

#define CREATE_ACCESSORS(x, type)                                                                  \
  enum sp_return sp_set_##x(struct sp_port *port, type x) {                                 \
    struct port_data data;                                                                         \
    struct sp_port_config config;                                                                  \
    {                                                                                              \
      enum sp_return ret = sp_validate_open_port(port, __func__);                                  \
      if (ret != SP_OK) return ret;                                                               \
    }                                                                                              \
    {                                                                                              \
      enum sp_return ret = get_config(port, &data, &config);                                       \
      if (ret != SP_OK) return sp_normalize_return(ret);                                           \
    }                                                                                              \
    config.x = x;                                                                                  \
    {                                                                                              \
      enum sp_return ret = set_config(port, &data, &config);                                       \
      if (ret != SP_OK) return sp_normalize_return(ret);                                           \
    }                                                                                              \
    return SP_OK;                                                                                   \
  }                                                                                                \
  enum sp_return sp_get_config_##x(const struct sp_port_config *config, type *x) {          \
    if (!x) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null result pointer");                                       \
    if (!config) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null config");                                          \
    *x = config->x;                                                                                \
    return SP_OK;                                                                                   \
  }                                                                                                \
  enum sp_return sp_set_config_##x(struct sp_port_config *config, type x) {                 \
    if (!config) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null config");                                          \
    config->x = x;                                                                                 \
    return SP_OK;                                                                                   \
  }

CREATE_ACCESSORS(baudrate, int)
CREATE_ACCESSORS(bits, int)
CREATE_ACCESSORS(parity, enum sp_parity)
CREATE_ACCESSORS(stopbits, int)
CREATE_ACCESSORS(rts, enum sp_rts)
CREATE_ACCESSORS(cts, enum sp_cts)
CREATE_ACCESSORS(dtr, enum sp_dtr)
CREATE_ACCESSORS(dsr, enum sp_dsr)
CREATE_ACCESSORS(xon_xoff, enum sp_xonxoff)

enum sp_return sp_set_config_flowcontrol(struct sp_port_config *config,
                                                enum sp_flowcontrol flowcontrol) {
  if (!config) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null configuration");

  if (flowcontrol > SP_FLOWCONTROL_DTRDSR) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Invalid flow control setting");

  if (flowcontrol == SP_FLOWCONTROL_XONXOFF) config->xon_xoff = SP_XONXOFF_INOUT;
  else config->xon_xoff = SP_XONXOFF_DISABLED;

  if (flowcontrol == SP_FLOWCONTROL_RTSCTS) {
    config->rts = SP_RTS_FLOW_CONTROL;
    config->cts = SP_CTS_FLOW_CONTROL;
  } else {
    if (config->rts == SP_RTS_FLOW_CONTROL) config->rts = SP_RTS_ON;
    config->cts = SP_CTS_IGNORE;
  }

  if (flowcontrol == SP_FLOWCONTROL_DTRDSR) {
    config->dtr = SP_DTR_FLOW_CONTROL;
    config->dsr = SP_DSR_FLOW_CONTROL;
  } else {
    if (config->dtr == SP_DTR_FLOW_CONTROL) config->dtr = SP_DTR_ON;
    config->dsr = SP_DSR_IGNORE;
  }

  return SP_OK;
}

enum sp_return sp_set_flowcontrol(struct sp_port *port, enum sp_flowcontrol flowcontrol) {
  struct port_data data;
  struct sp_port_config config;
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  {
    enum sp_return ret = get_config(port, &data, &config);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }

  {
    enum sp_return ret = sp_set_config_flowcontrol(&config, flowcontrol);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }

  {
    enum sp_return ret = set_config(port, &data, &config);
    if (ret != SP_OK) return sp_normalize_return(ret);
  }

  return SP_OK;
}

enum sp_return sp_get_signals(struct sp_port *port, enum sp_signal *signals) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }

  if (!signals) return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Null result pointer");

  TLOG_DEBUG("Getting control signals for port {}", port->name);

  *signals = 0;
#ifdef _WIN32
  DWORD bits;
  if (GetCommModemStatus(port->hdl, &bits) == 0) return sp_fail_return(__func__, "GetCommModemStatus() failed");
  if (bits & MS_CTS_ON) *signals |= SP_SIG_CTS;
  if (bits & MS_DSR_ON) *signals |= SP_SIG_DSR;
  if (bits & MS_RLSD_ON) *signals |= SP_SIG_DCD;
  if (bits & MS_RING_ON) *signals |= SP_SIG_RI;
#else
  int bits;
  if (ioctl(port->fd, TIOCMGET, &bits) < 0) return sp_fail_return(__func__, "TIOCMGET ioctl failed");
  if (bits & TIOCM_CTS) *signals |= SP_SIG_CTS;
  if (bits & TIOCM_DSR) *signals |= SP_SIG_DSR;
  if (bits & TIOCM_CAR) *signals |= SP_SIG_DCD;
  if (bits & TIOCM_RNG) *signals |= SP_SIG_RI;
#endif
  return SP_OK;
}

enum sp_return sp_start_break(struct sp_port *port) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }
#ifdef _WIN32
  if (SetCommBreak(port->hdl) == 0) return sp_fail_return(__func__, "SetCommBreak() failed");
#else
  if (ioctl(port->fd, TIOCSBRK, 1) < 0) return sp_fail_return(__func__, "TIOCSBRK ioctl failed");
#endif

  return SP_OK;
}

enum sp_return sp_end_break(struct sp_port *port) {
  {
    enum sp_return ret = sp_validate_open_port(port, __func__);
    if (ret != SP_OK) return ret;
  }
#ifdef _WIN32
  if (ClearCommBreak(port->hdl) == 0) return sp_fail_return(__func__, "ClearCommBreak() failed");
#else
  if (ioctl(port->fd, TIOCCBRK, 1) < 0) return sp_fail_return(__func__, "TIOCCBRK ioctl failed");
#endif

  return SP_OK;
}

int sp_last_error_code(void) {
#ifdef _WIN32
  return GetLastError();
#else
  return errno;
#endif
}

char *sp_last_error_message(void) {
#ifdef _WIN32
  char *message;
  DWORD error = GetLastError();

  DWORD length = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&message, 0, NULL);

  if (length >= 2 && message[length - 2] == '\r') message[length - 2] = '\0';

  return message;
#else
  return strerror(errno);
#endif
}

void sp_free_error_message(char *message) {
#ifdef _WIN32
  LocalFree(message);
#else
  (void)message;
#endif

  return;
}

int sp_get_major_package_version(void) { return SP_PACKAGE_VERSION_MAJOR; }

int sp_get_minor_package_version(void) { return SP_PACKAGE_VERSION_MINOR; }

int sp_get_micro_package_version(void) { return SP_PACKAGE_VERSION_MICRO; }

const char *sp_get_package_version_string(void) { return SP_PACKAGE_VERSION_STRING; }

int sp_get_current_lib_version(void) { return SP_LIB_VERSION_CURRENT; }

int sp_get_revision_lib_version(void) { return SP_LIB_VERSION_REVISION; }

int sp_get_age_lib_version(void) { return SP_LIB_VERSION_AGE; }

const char *sp_get_lib_version_string(void) { return SP_LIB_VERSION_STRING; }

/** @} */
