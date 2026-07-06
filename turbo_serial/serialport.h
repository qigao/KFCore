#ifndef TURBOSERIAL_SERIALPORT_H
#define TURBOSERIAL_SERIALPORT_H

#ifndef TURBOSERIAL_ATBUILD
#error "serialport.h is an internal libserialport implementation header"
#endif

#include "turbo_serial.h"


#ifdef __cplusplus
extern "C" {
#endif
/** Return values. */
enum sp_return {
  /** Operation completed successfully. */
  SP_OK = 0,
  /** Invalid arguments were passed to the function. */
  SP_ERR_ARG = -1,
  /** A system error occurred while executing the operation. */
  SP_ERR_FAIL = -2,
  /** A memory allocation failed while executing the operation. */
  SP_ERR_MEM = -3,
  /** The requested operation is not supported by this system or device. */
  SP_ERR_SUPP = -4
};

/** Port access modes. */
enum sp_mode {
  /** Open port for read access. */
  SP_MODE_READ = 1,
  /** Open port for write access. */
  SP_MODE_WRITE = 2,
  /** Open port for read and write access. @since 0.1.1 */
  SP_MODE_READ_WRITE = 3
};

/** Port events. */
enum sp_event {
  /** Data received and ready to read. */
  SP_EVENT_RX_READY = 1,
  /** Ready to transmit new data. */
  SP_EVENT_TX_READY = 2,
  /** Error occurred. */
  SP_EVENT_ERROR = 4
};

/** Buffer selection. */
enum sp_buffer {
  /** Input buffer. */
  SP_BUF_INPUT = 1,
  /** Output buffer. */
  SP_BUF_OUTPUT = 2,
  /** Both buffers. */
  SP_BUF_BOTH = 3
};

/** Parity settings. */
enum sp_parity {
  /** Special value to indicate setting should be left alone. */
  SP_PARITY_INVALID = -1,
  /** No parity. */
  SP_PARITY_NONE = 0,
  /** Odd parity. */
  SP_PARITY_ODD = 1,
  /** Even parity. */
  SP_PARITY_EVEN = 2,
  /** Mark parity. */
  SP_PARITY_MARK = 3,
  /** Space parity. */
  SP_PARITY_SPACE = 4
};

/** RTS pin behaviour. */
enum sp_rts {
  /** Special value to indicate setting should be left alone. */
  SP_RTS_INVALID = -1,
  /** RTS off. */
  SP_RTS_OFF = 0,
  /** RTS on. */
  SP_RTS_ON = 1,
  /** RTS used for flow control. */
  SP_RTS_FLOW_CONTROL = 2
};

/** CTS pin behaviour. */
enum sp_cts {
  /** Special value to indicate setting should be left alone. */
  SP_CTS_INVALID = -1,
  /** CTS ignored. */
  SP_CTS_IGNORE = 0,
  /** CTS used for flow control. */
  SP_CTS_FLOW_CONTROL = 1
};

/** DTR pin behaviour. */
enum sp_dtr {
  /** Special value to indicate setting should be left alone. */
  SP_DTR_INVALID = -1,
  /** DTR off. */
  SP_DTR_OFF = 0,
  /** DTR on. */
  SP_DTR_ON = 1,
  /** DTR used for flow control. */
  SP_DTR_FLOW_CONTROL = 2
};

/** DSR pin behaviour. */
enum sp_dsr {
  /** Special value to indicate setting should be left alone. */
  SP_DSR_INVALID = -1,
  /** DSR ignored. */
  SP_DSR_IGNORE = 0,
  /** DSR used for flow control. */
  SP_DSR_FLOW_CONTROL = 1
};

/** XON/XOFF flow control behaviour. */
enum sp_xonxoff {
  /** Special value to indicate setting should be left alone. */
  SP_XONXOFF_INVALID = -1,
  /** XON/XOFF disabled. */
  SP_XONXOFF_DISABLED = 0,
  /** XON/XOFF enabled for input only. */
  SP_XONXOFF_IN = 1,
  /** XON/XOFF enabled for output only. */
  SP_XONXOFF_OUT = 2,
  /** XON/XOFF enabled for input and output. */
  SP_XONXOFF_INOUT = 3
};

/** Standard flow control combinations. */
enum sp_flowcontrol {
  /** No flow control. */
  SP_FLOWCONTROL_NONE = 0,
  /** Software flow control using XON/XOFF characters. */
  SP_FLOWCONTROL_XONXOFF = 1,
  /** Hardware flow control using RTS/CTS signals. */
  SP_FLOWCONTROL_RTSCTS = 2,
  /** Hardware flow control using DTR/DSR signals. */
  SP_FLOWCONTROL_DTRDSR = 3
};

/** Input signals. */
enum sp_signal {
  /** Clear to send. */
  SP_SIG_CTS = 1,
  /** Data set ready. */
  SP_SIG_DSR = 2,
  /** Data carrier detect. */
  SP_SIG_DCD = 4,
  /** Ring indicator. */
  SP_SIG_RI = 8
};

/**
 * Transport types.
 *
 * @since 0.1.1
 */
enum sp_transport {
  /** Native platform serial port. @since 0.1.1 */
  SP_TRANSPORT_NATIVE,
  /** USB serial port adapter. @since 0.1.1 */
  SP_TRANSPORT_USB,
  /** Bluetooth serial port adapter. @since 0.1.1 */
  SP_TRANSPORT_BLUETOOTH
};

/**
 * @struct sp_port
 * An opaque structure representing a serial port.
 */
struct sp_port;

/**
 * @struct sp_port_config
 * An opaque structure representing the configuration for a serial port.
 */
struct sp_port_config;

/**
 * @struct sp_event_set
 * A set of handles to wait on for events.
 */
struct sp_event_set {
  /** Array of OS-specific handles. */
  void *handles;
  /** Array of bitmasks indicating which events apply for each handle. */
  enum sp_event *masks;
  /** Number of handles. */
  unsigned int count;
};

/**
 * @defgroup Enumeration Port enumeration
 *
 * Enumerating the serial ports of a system.
 *
 * See @ref list_ports.c for a working example of port enumeration.
 *
 * @{
 */

/**
 * Obtain a pointer to a new sp_port structure representing the named port.
 *
 * The user should allocate a variable of type "struct sp_port *" and pass a
 * pointer to this to receive the result.
 *
 * The result should be freed after use by calling sp_free_port().
 *
 * @param[in] portname The OS-specific name of a serial port. Must not be NULL.
 * @param[out] port_ptr If any error is returned, the variable pointed to by
 *                      port_ptr will be set to NULL. Otherwise, it will be set
 *                      to point to the newly allocated port. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_port_by_name(const char *portname, struct sp_port **port_ptr);

/**
 * Free a port structure obtained from sp_get_port_by_name() or sp_copy_port().
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @since 0.1.0
 */
void sp_free_port(struct sp_port *port);

/**
 * List the serial ports available on the system.
 *
 * The result obtained is an array of pointers to sp_port structures,
 * terminated by a NULL. The user should allocate a variable of type
 * "struct sp_port **" and pass a pointer to this to receive the result.
 *
 * The result should be freed after use by calling sp_free_port_list().
 * If a port from the list is to be used after freeing the list, it must be
 * copied first using sp_copy_port().
 *
 * @param[out] list_ptr If any error is returned, the variable pointed to by
 *                      list_ptr will be set to NULL. Otherwise, it will be set
 *                      to point to the newly allocated array. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_list_ports(struct sp_port ***list_ptr);

/**
 * Make a new copy of an sp_port structure.
 *
 * The user should allocate a variable of type "struct sp_port *" and pass a
 * pointer to this to receive the result.
 *
 * The copy should be freed after use by calling sp_free_port().
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[out] copy_ptr If any error is returned, the variable pointed to by
 *                      copy_ptr will be set to NULL. Otherwise, it will be set
 *                      to point to the newly allocated copy. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_copy_port(const struct sp_port *port, struct sp_port **copy_ptr);

/**
 * Free a port list obtained from sp_list_ports().
 *
 * This will also free all the sp_port structures referred to from the list;
 * any that are to be retained must be copied first using sp_copy_port().
 *
 * @param[in] ports Pointer to a list of port structures. Must not be NULL.
 *
 * @since 0.1.0
 */
void sp_free_port_list(struct sp_port **ports);

/**
 * @}
 * @defgroup Ports Port handling
 *
 * Opening, closing and querying ports.
 *
 * See @ref port_info.c for a working example of getting port information.
 *
 * @{
 */

/**
 * Open the specified serial port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] flags Flags to use when opening the serial port.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_open(struct sp_port *port, enum sp_mode flags);

/**
 * Close the specified serial port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_close(struct sp_port *port);

/**
 * Get the name of a port.
 *
 * The name returned is whatever is normally used to refer to a port on the
 * current operating system; e.g. for Windows it will usually be a "COMn"
 * device name, and for Unix it will be a device path beginning with "/dev/".
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return The port name, or NULL if an invalid port is passed. The name
 *         string is part of the port structure and may not be used after
 *         the port structure has been freed.
 *
 * @since 0.1.0
 */
char *sp_get_port_name(const struct sp_port *port);

/**
 * Get a description for a port, to present to end user.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return The port description, or NULL if an invalid port is passed.
 *         The description string is part of the port structure and may not
 *         be used after the port structure has been freed.
 *
 * @since 0.1.1
 */
char *sp_get_port_description(const struct sp_port *port);

/**
 * Get the transport type used by a port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return The port transport type.
 *
 * @since 0.1.1
 */
enum sp_transport sp_get_port_transport(const struct sp_port *port);

/**
 * Get the USB bus number and address on bus of a USB serial adapter port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[out] usb_bus Pointer to a variable to store the USB bus.
 *                     Can be NULL (in that case it will be ignored).
 * @param[out] usb_address Pointer to a variable to store the USB address.
 *                         Can be NULL (in that case it will be ignored).
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.1
 */
enum sp_return sp_get_port_usb_bus_address(const struct sp_port *port, int *usb_bus,
                                                  int *usb_address);

/**
 * Get the USB Vendor ID and Product ID of a USB serial adapter port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[out] usb_vid Pointer to a variable to store the USB VID.
 *                     Can be NULL (in that case it will be ignored).
 * @param[out] usb_pid Pointer to a variable to store the USB PID.
 *                     Can be NULL (in that case it will be ignored).
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.1
 */
enum sp_return sp_get_port_usb_vid_pid(const struct sp_port *port, int *usb_vid,
                                              int *usb_pid);

/**
 * Get the USB manufacturer string of a USB serial adapter port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return The port manufacturer string, or NULL if an invalid port is passed.
 *         The manufacturer string is part of the port structure and may not
 *         be used after the port structure has been freed.
 *
 * @since 0.1.1
 */
char *sp_get_port_usb_manufacturer(const struct sp_port *port);

/**
 * Get the USB product string of a USB serial adapter port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return The port product string, or NULL if an invalid port is passed.
 *         The product string is part of the port structure and may not be
 *         used after the port structure has been freed.
 *
 * @since 0.1.1
 */
char *sp_get_port_usb_product(const struct sp_port *port);

/**
 * Get the USB serial number string of a USB serial adapter port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return The port serial number, or NULL if an invalid port is passed.
 *         The serial number string is part of the port structure and may
 *         not be used after the port structure has been freed.
 *
 * @since 0.1.1
 */
char *sp_get_port_usb_serial(const struct sp_port *port);

/**
 * Get the MAC address of a Bluetooth serial adapter port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return The port MAC address, or NULL if an invalid port is passed.
 *         The MAC address string is part of the port structure and may not
 *         be used after the port structure has been freed.
 *
 * @since 0.1.1
 */
char *sp_get_port_bluetooth_address(const struct sp_port *port);

/**
 * Get the operating system handle for a port.
 *
 * The type of the handle depends on the operating system. On Unix based
 * systems, the handle is a file descriptor of type "int". On Windows, the
 * handle is of type "HANDLE". The user should allocate a variable of the
 * appropriate type and pass a pointer to this to receive the result.
 *
 * To obtain a valid handle, the port must first be opened by calling
 * sp_open() using the same port structure.
 *
 * After the port is closed or the port structure freed, the handle may
 * no longer be valid.
 *
 * @warning This feature is provided so that programs may make use of
 *          OS-specific functionality where desired. Doing so obviously
 *          comes at a cost in portability. It also cannot be guaranteed
 *          that direct usage of the OS handle will not conflict with the
 *          library's own usage of the port. Be careful.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[out] result_ptr If any error is returned, the variable pointed to by
 *                        result_ptr will have unknown contents and should not
 *                        be used. Otherwise, it will be set to point to the
 *                        OS handle. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_port_handle(const struct sp_port *port, void *result_ptr);

/**
 * @}
 *
 * @defgroup Configuration Configuration
 *
 * Setting and querying serial port parameters.
 *
 * See @ref port_config.c for a working example of port configuration.
 *
 * You should always configure all settings before using a port.
 * There are no default settings applied by libserialport.
 * When you open a port it may have default settings from the OS or
 * driver, or the settings left over by the last program to use it.
 *
 * You should always set baud rate, data bits, parity and stop bits.
 *
 * You should normally also set one of the preset @ref sp_flowcontrol
 * flow control modes, which will set up the RTS, CTS, DTR and DSR pin
 * behaviours and enable or disable XON/XOFF. If you need an unusual
 * configuration not covered by the preset flow control modes, you
 * will need to configure these settings individually, and avoid
 * calling sp_set_flowcontrol() or sp_set_config_flowcontrol() which
 * will overwrite these settings.
 *
 * A port must be opened before you can change its settings.
 *
 * There are two ways of accessing port settings:
 *
 * Configuration structures
 * ------------------------
 *
 * You can read and write a whole configuration (all settings at once)
 * using sp_get_config() and sp_set_config(). This is handy if you want
 * to change between some preset combinations, or save and restore an
 * existing configuration. It also ensures the changes are made
 * together, via an efficient set of calls into the OS - in some cases
 * a single system call can be used.
 *
 * Use accessor functions like sp_get_config_baudrate() and
 * sp_set_config_baudrate() to get and set individual settings
 * from a configuration.
 *
 * For each setting in a port configuration, a special value of -1 can
 * be used, which will cause that setting to be left alone when the
 * configuration is applied by sp_set_config().
 *
 * This value is also be used by sp_get_config() for any settings
 * which are unconfigured at the OS level, or in a state that is
 * not representable within the libserialport API.
 *
 * Configurations are allocated using sp_new_config() and freed
 * with sp_free_config(). You need to manage them yourself. When
 * a new configuration is allocated by sp_new_config(), all of
 * its settings are initially set to the special -1 value.
 *
 * Direct functions for changing port settings
 * -------------------------------------------
 *
 * As a shortcut, you can set individual settings on a port directly
 * by calling functions like sp_set_baudrate() and sp_set_parity().
 * This saves you the work of allocating a temporary config, setting it
 * up, applying it to a port and then freeing it.
 *
 * @{
 */

/**
 * Allocate a port configuration structure.
 *
 * The user should allocate a variable of type "struct sp_port_config *" and
 * pass a pointer to this to receive the result. The variable will be updated
 * to point to the new configuration structure. The structure is opaque and
 * must be accessed via the functions provided.
 *
 * All parameters in the structure will be initialised to special values which
 * are ignored by sp_set_config().
 *
 * The structure should be freed after use by calling sp_free_config().
 *
 * @param[out] config_ptr If any error is returned, the variable pointed to by
 *                        config_ptr will be set to NULL. Otherwise, it will
 *                        be set to point to the allocated config structure.
 *                        Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_new_config(struct sp_port_config **config_ptr);

/**
 * Free a port configuration structure.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 *
 * @since 0.1.0
 */
void sp_free_config(struct sp_port_config *config);

/**
 * Get the current configuration of the specified serial port.
 *
 * The user should allocate a configuration structure using sp_new_config()
 * and pass this as the config parameter. The configuration structure will
 * be updated with the port configuration.
 *
 * Any parameters that are configured with settings not recognised or
 * supported by libserialport, will be set to special values that are
 * ignored by sp_set_config().
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[out] config Pointer to a configuration structure that will hold
 *                    the result. Upon errors the contents of the config
 *                    struct will not be changed. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_config(struct sp_port *port, struct sp_port_config *config);

/**
 * Set the configuration for the specified serial port.
 *
 * For each parameter in the configuration, there is a special value (usually
 * -1, but see the documentation for each field). These values will be ignored
 * and the corresponding setting left unchanged on the port.
 *
 * Upon errors, the configuration of the serial port is unknown since
 * partial/incomplete config updates may have happened.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_config(struct sp_port *port, const struct sp_port_config *config);

/**
 * Set the baud rate for the specified serial port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] baudrate Baud rate in bits per second.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_baudrate(struct sp_port *port, int baudrate);

/**
 * Get the baud rate from a port configuration.
 *
 * The user should allocate a variable of type int and
 * pass a pointer to this to receive the result.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[out] baudrate_ptr Pointer to a variable to store the result. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_config_baudrate(const struct sp_port_config *config,
                                             int *baudrate_ptr);

/**
 * Set the baud rate in a port configuration.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[in] baudrate Baud rate in bits per second, or -1 to retain the current setting.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_config_baudrate(struct sp_port_config *config, int baudrate);

/**
 * Set the data bits for the specified serial port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] bits Number of data bits.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_bits(struct sp_port *port, int bits);

/**
 * Get the data bits from a port configuration.
 *
 * The user should allocate a variable of type int and
 * pass a pointer to this to receive the result.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[out] bits_ptr Pointer to a variable to store the result. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_config_bits(const struct sp_port_config *config, int *bits_ptr);

/**
 * Set the data bits in a port configuration.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[in] bits Number of data bits, or -1 to retain the current setting.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_config_bits(struct sp_port_config *config, int bits);

/**
 * Set the parity setting for the specified serial port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] parity Parity setting.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_parity(struct sp_port *port, enum sp_parity parity);

/**
 * Get the parity setting from a port configuration.
 *
 * The user should allocate a variable of type enum sp_parity and
 * pass a pointer to this to receive the result.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[out] parity_ptr Pointer to a variable to store the result. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_config_parity(const struct sp_port_config *config,
                                           enum sp_parity *parity_ptr);

/**
 * Set the parity setting in a port configuration.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[in] parity Parity setting, or -1 to retain the current setting.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_config_parity(struct sp_port_config *config, enum sp_parity parity);

/**
 * Set the stop bits for the specified serial port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] stopbits Number of stop bits.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_stopbits(struct sp_port *port, int stopbits);

/**
 * Get the stop bits from a port configuration.
 *
 * The user should allocate a variable of type int and
 * pass a pointer to this to receive the result.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[out] stopbits_ptr Pointer to a variable to store the result. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_config_stopbits(const struct sp_port_config *config,
                                             int *stopbits_ptr);

/**
 * Set the stop bits in a port configuration.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[in] stopbits Number of stop bits, or -1 to retain the current setting.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_config_stopbits(struct sp_port_config *config, int stopbits);

/**
 * Set the RTS pin behaviour for the specified serial port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] rts RTS pin mode.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_rts(struct sp_port *port, enum sp_rts rts);

/**
 * Get the RTS pin behaviour from a port configuration.
 *
 * The user should allocate a variable of type enum sp_rts and
 * pass a pointer to this to receive the result.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[out] rts_ptr Pointer to a variable to store the result. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_config_rts(const struct sp_port_config *config, enum sp_rts *rts_ptr);

/**
 * Set the RTS pin behaviour in a port configuration.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[in] rts RTS pin mode, or -1 to retain the current setting.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_config_rts(struct sp_port_config *config, enum sp_rts rts);

/**
 * Set the CTS pin behaviour for the specified serial port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] cts CTS pin mode.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_cts(struct sp_port *port, enum sp_cts cts);

/**
 * Get the CTS pin behaviour from a port configuration.
 *
 * The user should allocate a variable of type enum sp_cts and
 * pass a pointer to this to receive the result.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[out] cts_ptr Pointer to a variable to store the result. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_config_cts(const struct sp_port_config *config, enum sp_cts *cts_ptr);

/**
 * Set the CTS pin behaviour in a port configuration.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[in] cts CTS pin mode, or -1 to retain the current setting.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_config_cts(struct sp_port_config *config, enum sp_cts cts);

/**
 * Set the DTR pin behaviour for the specified serial port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] dtr DTR pin mode.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_dtr(struct sp_port *port, enum sp_dtr dtr);

/**
 * Get the DTR pin behaviour from a port configuration.
 *
 * The user should allocate a variable of type enum sp_dtr and
 * pass a pointer to this to receive the result.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[out] dtr_ptr Pointer to a variable to store the result. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_config_dtr(const struct sp_port_config *config, enum sp_dtr *dtr_ptr);

/**
 * Set the DTR pin behaviour in a port configuration.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[in] dtr DTR pin mode, or -1 to retain the current setting.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_config_dtr(struct sp_port_config *config, enum sp_dtr dtr);

/**
 * Set the DSR pin behaviour for the specified serial port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] dsr DSR pin mode.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_dsr(struct sp_port *port, enum sp_dsr dsr);

/**
 * Get the DSR pin behaviour from a port configuration.
 *
 * The user should allocate a variable of type enum sp_dsr and
 * pass a pointer to this to receive the result.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[out] dsr_ptr Pointer to a variable to store the result. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_config_dsr(const struct sp_port_config *config, enum sp_dsr *dsr_ptr);

/**
 * Set the DSR pin behaviour in a port configuration.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[in] dsr DSR pin mode, or -1 to retain the current setting.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_config_dsr(struct sp_port_config *config, enum sp_dsr dsr);

/**
 * Set the XON/XOFF configuration for the specified serial port.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] xon_xoff XON/XOFF mode.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_xon_xoff(struct sp_port *port, enum sp_xonxoff xon_xoff);

/**
 * Get the XON/XOFF configuration from a port configuration.
 *
 * The user should allocate a variable of type enum sp_xonxoff and
 * pass a pointer to this to receive the result.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[out] xon_xoff_ptr Pointer to a variable to store the result. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_config_xon_xoff(const struct sp_port_config *config,
                                             enum sp_xonxoff *xon_xoff_ptr);

/**
 * Set the XON/XOFF configuration in a port configuration.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[in] xon_xoff XON/XOFF mode, or -1 to retain the current setting.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_config_xon_xoff(struct sp_port_config *config,
                                             enum sp_xonxoff xon_xoff);

/**
 * Set the flow control type in a port configuration.
 *
 * This function is a wrapper that sets the RTS, CTS, DTR, DSR and
 * XON/XOFF settings as necessary for the specified flow control
 * type. For more fine-grained control of these settings, use their
 * individual configuration functions.
 *
 * @param[in] config Pointer to a configuration structure. Must not be NULL.
 * @param[in] flowcontrol Flow control setting to use.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_config_flowcontrol(struct sp_port_config *config,
                                                enum sp_flowcontrol flowcontrol);

/**
 * Set the flow control type for the specified serial port.
 *
 * This function is a wrapper that sets the RTS, CTS, DTR, DSR and
 * XON/XOFF settings as necessary for the specified flow control
 * type. For more fine-grained control of these settings, use their
 * individual configuration functions.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] flowcontrol Flow control setting to use.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_set_flowcontrol(struct sp_port *port, enum sp_flowcontrol flowcontrol);

/**
 * @}
 *
 * @defgroup Data Data handling
 *
 * Reading, writing, and flushing data.
 *
 * See @ref send_receive.c for an example of sending and receiving data.
 *
 * @{
 */

/**
 * Read bytes from the specified serial port, blocking until complete.
 *
 * @warning If your program runs on Unix, defines its own signal handlers, and
 *          needs to abort blocking reads when these are called, then you
 *          should not use this function. It repeats system calls that return
 *          with EINTR. To be able to abort a read from a signal handler, you
 *          should implement your own blocking read using sp_nonblocking_read()
 *          together with a blocking method that makes sense for your program.
 *          E.g. you can obtain the file descriptor for an open port using
 *          sp_get_port_handle() and use this to call select() or pselect(),
 *          with appropriate arrangements to return if a signal is received.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[out] buf Buffer in which to store the bytes read. Must not be NULL.
 * @param[in] count Requested number of bytes to read.
 * @param[in] timeout_ms Timeout in milliseconds, or zero to wait indefinitely.
 *
 * @return The number of bytes read on success, or a negative error code. If
 *         the number of bytes returned is less than that requested, the
 *         timeout was reached before the requested number of bytes was
 *         available. If timeout is zero, the function will always return
 *         either the requested number of bytes or a negative error code.
 *
 * @since 0.1.0
 */
enum sp_return sp_blocking_read(struct sp_port *port, void *buf, size_t count,
                                       unsigned int timeout_ms);

/**
 * Read bytes from the specified serial port, returning as soon as any data is
 * available.
 *
 * @warning If your program runs on Unix, defines its own signal handlers, and
 *          needs to abort blocking reads when these are called, then you
 *          should not use this function. It repeats system calls that return
 *          with EINTR. To be able to abort a read from a signal handler, you
 *          should implement your own blocking read using sp_nonblocking_read()
 *          together with a blocking method that makes sense for your program.
 *          E.g. you can obtain the file descriptor for an open port using
 *          sp_get_port_handle() and use this to call select() or pselect(),
 *          with appropriate arrangements to return if a signal is received.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[out] buf Buffer in which to store the bytes read. Must not be NULL.
 * @param[in] count Maximum number of bytes to read. Must not be zero.
 * @param[in] timeout_ms Timeout in milliseconds, or zero to wait indefinitely.
 *
 * @return The number of bytes read on success, or a negative error code. If
 *         the result is zero, the timeout was reached before any bytes were
 *         available. If timeout_ms is zero, the function will always return
 *         either at least one byte, or a negative error code.
 *
 * @since 0.1.1
 */
enum sp_return sp_blocking_read_next(struct sp_port *port, void *buf, size_t count,
                                            unsigned int timeout_ms);

/**
 * Read bytes from the specified serial port, without blocking.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[out] buf Buffer in which to store the bytes read. Must not be NULL.
 * @param[in] count Maximum number of bytes to read.
 *
 * @return The number of bytes read on success, or a negative error code. The
 *         number of bytes returned may be any number from zero to the maximum
 *         that was requested.
 *
 * @since 0.1.0
 */
enum sp_return sp_nonblocking_read(struct sp_port *port, void *buf, size_t count);

/**
 * Write bytes to the specified serial port, blocking until complete.
 *
 * Note that this function only ensures that the accepted bytes have been
 * written to the OS; they may be held in driver or hardware buffers and not
 * yet physically transmitted. To check whether all written bytes have actually
 * been transmitted, use the sp_output_waiting() function. To wait until all
 * written bytes have actually been transmitted, use the sp_drain() function.
 *
 * @warning If your program runs on Unix, defines its own signal handlers, and
 *          needs to abort blocking writes when these are called, then you
 *          should not use this function. It repeats system calls that return
 *          with EINTR. To be able to abort a write from a signal handler, you
 *          should implement your own blocking write using sp_nonblocking_write()
 *          together with a blocking method that makes sense for your program.
 *          E.g. you can obtain the file descriptor for an open port using
 *          sp_get_port_handle() and use this to call select() or pselect(),
 *          with appropriate arrangements to return if a signal is received.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] buf Buffer containing the bytes to write. Must not be NULL.
 * @param[in] count Requested number of bytes to write.
 * @param[in] timeout_ms Timeout in milliseconds, or zero to wait indefinitely.
 *
 * @return The number of bytes written on success, or a negative error code.
 *         If the number of bytes returned is less than that requested, the
 *         timeout was reached before the requested number of bytes was
 *         written. If timeout is zero, the function will always return
 *         either the requested number of bytes or a negative error code. In
 *         the event of an error there is no way to determine how many bytes
 *         were sent before the error occurred.
 *
 * @since 0.1.0
 */
enum sp_return sp_blocking_write(struct sp_port *port, const void *buf, size_t count,
                                        unsigned int timeout_ms);

/**
 * Write bytes to the specified serial port, without blocking.
 *
 * Note that this function only ensures that the accepted bytes have been
 * written to the OS; they may be held in driver or hardware buffers and not
 * yet physically transmitted. To check whether all written bytes have actually
 * been transmitted, use the sp_output_waiting() function. To wait until all
 * written bytes have actually been transmitted, use the sp_drain() function.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] buf Buffer containing the bytes to write. Must not be NULL.
 * @param[in] count Maximum number of bytes to write.
 *
 * @return The number of bytes written on success, or a negative error code.
 *         The number of bytes returned may be any number from zero to the
 *         maximum that was requested.
 *
 * @since 0.1.0
 */
enum sp_return sp_nonblocking_write(struct sp_port *port, const void *buf, size_t count);

/**
 * Gets the number of bytes waiting in the input buffer.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return Number of bytes waiting on success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_input_waiting(struct sp_port *port);

/**
 * Gets the number of bytes waiting in the output buffer.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return Number of bytes waiting on success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_output_waiting(struct sp_port *port);

/**
 * Flush serial port buffers. Data in the selected buffer(s) is discarded.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] buffers Which buffer(s) to flush.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_flush(struct sp_port *port, enum sp_buffer buffers);

/**
 * Wait for buffered data to be transmitted.
 *
 * @warning If your program runs on Unix, defines its own signal handlers, and
 *          needs to abort draining the output buffer when when these are
 *          called, then you should not use this function. It repeats system
 *          calls that return with EINTR. To be able to abort a drain from a
 *          signal handler, you would need to implement your own blocking
 *          drain by polling the result of sp_output_waiting().
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_drain(struct sp_port *port);

/**
 * @}
 *
 * @defgroup Waiting Waiting
 *
 * Waiting for events and timeout handling.
 *
 * See @ref await_events.c for an example of awaiting events on multiple ports.
 *
 * @{
 */

/**
 * Allocate storage for a set of events.
 *
 * The user should allocate a variable of type struct sp_event_set *,
 * then pass a pointer to this variable to receive the result.
 *
 * The result should be freed after use by calling sp_free_event_set().
 *
 * @param[out] result_ptr If any error is returned, the variable pointed to by
 *                        result_ptr will be set to NULL. Otherwise, it will
 *                        be set to point to the event set. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_new_event_set(struct sp_event_set **result_ptr);

/**
 * Add events to a struct sp_event_set for a given port.
 *
 * The port must first be opened by calling sp_open() using the same port
 * structure.
 *
 * After the port is closed or the port structure freed, the results may
 * no longer be valid.
 *
 * @param[in,out] event_set Event set to update. Must not be NULL.
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[in] mask Bitmask of events to be waited for.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_add_port_events(struct sp_event_set *event_set, const struct sp_port *port,
                                         enum sp_event mask);

/**
 * Wait for any of a set of events to occur.
 *
 * @param[in] event_set Event set to wait on. Must not be NULL.
 * @param[in] timeout_ms Timeout in milliseconds, or zero to wait indefinitely.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_wait(struct sp_event_set *event_set, unsigned int timeout_ms);

/**
 * Free a structure allocated by sp_new_event_set().
 *
 * @param[in] event_set Event set to free. Must not be NULL.
 *
 * @since 0.1.0
 */
void sp_free_event_set(struct sp_event_set *event_set);

/**
 * @}
 *
 * @defgroup Signals Signals
 *
 * Port signalling operations.
 *
 * @{
 */

/**
 * Gets the status of the control signals for the specified port.
 *
 * The user should allocate a variable of type "enum sp_signal" and pass a
 * pointer to this variable to receive the result. The result is a bitmask
 * in which individual signals can be checked by bitwise OR with values of
 * the sp_signal enum.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 * @param[out] signal_mask Pointer to a variable to receive the result.
 *                         Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_get_signals(struct sp_port *port, enum sp_signal *signal_mask);

/**
 * Put the port transmit line into the break state.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_start_break(struct sp_port *port);

/**
 * Take the port transmit line out of the break state.
 *
 * @param[in] port Pointer to a port structure. Must not be NULL.
 *
 * @return SP_OK upon success, a negative error code otherwise.
 *
 * @since 0.1.0
 */
enum sp_return sp_end_break(struct sp_port *port);

/**
 * @}
 *
 * @defgroup Errors Errors
 *
 * Obtaining error information.
 *
 * See @ref handle_errors.c for an example of error handling.
 *
 * @{
 */

/**
 * Get the error code for a failed operation.
 *
 * In order to obtain the correct result, this function should be called
 * straight after the failure, before executing any other system operations.
 * The result is thread-specific, and only valid when called immediately
 * after a previous call returning SP_ERR_FAIL.
 *
 * @return The system's numeric code for the error that caused the last
 *         operation to fail.
 *
 * @since 0.1.0
 */
int sp_last_error_code(void);

/**
 * Get the error message for a failed operation.
 *
 * In order to obtain the correct result, this function should be called
 * straight after the failure, before executing other system operations.
 * The result is thread-specific, and only valid when called immediately
 * after a previous call returning SP_ERR_FAIL.
 *
 * @return The system's message for the error that caused the last
 *         operation to fail. This string may be allocated by the function,
 *         and should be freed after use by calling sp_free_error_message().
 *
 * @since 0.1.0
 */
char *sp_last_error_message(void);

/**
 * Free an error message returned by sp_last_error_message().
 *
 * @param[in] message The error message string to free. Must not be NULL.
 *
 * @since 0.1.0
 */
void sp_free_error_message(char *message);

/** @} */

/**
 * @defgroup Versions Versions
 *
 * Version number querying functions, definitions, and macros.
 *
 * This set of API calls returns two different version numbers related
 * to libserialport. The "package version" is the release version number of the
 * libserialport tarball in the usual "major.minor.micro" format, e.g. "0.1.0".
 *
 * The "library version" is independent of that; it is the libtool version
 * number in the "current:revision:age" format, e.g. "2:0:0".
 * See http://www.gnu.org/software/libtool/manual/libtool.html#Libtool-versioning for details.
 *
 * Both version numbers (and/or individual components of them) can be
 * retrieved via the API calls at runtime, and/or they can be checked at
 * compile/preprocessor time using the respective macros.
 *
 * @{
 */

/*
 * Package version macros (can be used for conditional compilation).
 */

/** The libserialport package 'major' version number. */
#define SP_PACKAGE_VERSION_MAJOR 0

/** The libserialport package 'minor' version number. */
#define SP_PACKAGE_VERSION_MINOR 1

/** The libserialport package 'micro' version number. */
#define SP_PACKAGE_VERSION_MICRO 2

/** The libserialport package version ("major.minor.micro") as string. */
#define SP_PACKAGE_VERSION_STRING "0.1.2"

/*
 * Library/libtool version macros (can be used for conditional compilation).
 */

/** The libserialport libtool 'current' version number. */
#define SP_LIB_VERSION_CURRENT 1

/** The libserialport libtool 'revision' version number. */
#define SP_LIB_VERSION_REVISION 1

/** The libserialport libtool 'age' version number. */
#define SP_LIB_VERSION_AGE 1

/** The libserialport libtool version ("current:revision:age") as string. */
#define SP_LIB_VERSION_STRING "1:1:1"

/**
 * Get the major libserialport package version number.
 *
 * @return The major package version number.
 *
 * @since 0.1.0
 */
int sp_get_major_package_version(void);

/**
 * Get the minor libserialport package version number.
 *
 * @return The minor package version number.
 *
 * @since 0.1.0
 */
int sp_get_minor_package_version(void);

/**
 * Get the micro libserialport package version number.
 *
 * @return The micro package version number.
 *
 * @since 0.1.0
 */
int sp_get_micro_package_version(void);

/**
 * Get the libserialport package version number as a string.
 *
 * @return The package version number string. The returned string is
 *         static and thus should NOT be free'd by the caller.
 *
 * @since 0.1.0
 */
const char *sp_get_package_version_string(void);

/**
 * Get the "current" part of the libserialport library version number.
 *
 * @return The "current" library version number.
 *
 * @since 0.1.0
 */
int sp_get_current_lib_version(void);

/**
 * Get the "revision" part of the libserialport library version number.
 *
 * @return The "revision" library version number.
 *
 * @since 0.1.0
 */
int sp_get_revision_lib_version(void);

/**
 * Get the "age" part of the libserialport library version number.
 *
 * @return The "age" library version number.
 *
 * @since 0.1.0
 */
int sp_get_age_lib_version(void);

/**
 * Get the libserialport library version number as a string.
 *
 * @return The library version number string. The returned string is
 *         static and thus should NOT be free'd by the caller.
 *
 * @since 0.1.0
 */
const char *sp_get_lib_version_string(void);

/** @} */

/**
 * @example list_ports.c Getting a list of ports present on the system.
 * @example port_info.c Getting information on a particular serial port.
 * @example port_config.c Accessing configuration settings of a port.
 * @example send_receive.c Sending and receiving data.
 * @example await_events.c Awaiting events on multiple ports.
 * @example handle_errors.c Handling errors returned from the library.
 */


/* These MSVC-specific defines must appear before other headers.*/
#ifdef _MSC_VER
#define _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifndef TURBOSERIAL_API_ONLY

/* These feature test macros must appear before other headers.*/
#if defined(__linux__) || defined(__CYGWIN__)
/* For realpath. */
#define _BSD_SOURCE 1 /* for glibc < 2.19 */
#define _DEFAULT_SOURCE 1 /* for glibc >= 2.20 */
#define _POSIX_C_SOURCE 199309L
#endif

#ifdef TURBOSERIAL_ATBUILD
/* If building with autoconf, include the generated config.h. */
#include <config.h>
#endif

#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include "tlog.h"
#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#include <setupapi.h>
#include <cfgmgr32.h>
#include <winioctl.h>
/*
 * These are Windows SetupAPI GUID constants, not project/business UUID values.
 * Keep them as Windows GUID so calls such as SetupDiGetClassDevs() receive the
 * ABI type expected by the OS. TurboScript's vendor/uuid remains the source for
 * data-model UUID parsing, formatting and generation.
 */
#undef DEFINE_GUID
#define DEFINE_GUID(name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8) \
	static const GUID name = { l,w1,w2,{ b1,b2,b3,b4,b5,b6,b7,b8 } }
#include <usbioctl.h>
#include <usbiodef.h>
/* The largest size that can be passed to WriteFile() safely
 * on any architecture. This arises from the expression:
 * PAGE_SIZE * (65535 - sizeof(MDL)) / sizeof(ULONG_PTR)
 * and this worst-case value is found on x64. */
#define WRITEFILE_MAX_SIZE 33525760
#else
#include <limits.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <time.h>
#include <poll.h>
#include <unistd.h>
#ifdef HAVE_SYS_FILE_H
#include <sys/file.h>
#endif
#endif
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/serial/IOSerialKeys.h>
#include <IOKit/serial/ioss.h>
#include <sys/syslimits.h>
#include <mach/mach_time.h>
#if MAC_OS_X_VERSION_MAX_ALLOWED < 120000 /* Before macOS 12 */
#define kIOMainPortDefault kIOMasterPortDefault
#endif
#endif
#ifdef __linux__
#include <dirent.h>
/* Android only has linux/serial.h from platform 21 onwards. */
#if !(defined(__ANDROID__) && (__ANDROID_API__ < 21))
#include <linux/serial.h>
#endif
#include "linux_termios.h"

/* TCGETX/TCSETX is not available everywhere. */
#if defined(TCGETX) && defined(TCSETX) && defined(HAVE_STRUCT_TERMIOX)
#define USE_TERMIOX
#endif
#endif

/* TIOCINQ/TIOCOUTQ is not available everywhere. */
#if !defined(TIOCINQ) && defined(FIONREAD)
#define TIOCINQ FIONREAD
#endif
#if !defined(TIOCOUTQ) && defined(FIONWRITE)
#define TIOCOUTQ FIONWRITE
#endif

/*
 * O_CLOEXEC is not available everywhere, fallback to not setting the
 * flag on those systems.
 */
#ifndef _WIN32
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#endif

/* Non-standard baudrates are not available everywhere. */
#if (defined(HAVE_TERMIOS_SPEED) || defined(HAVE_TERMIOS2_SPEED)) && HAVE_DECL_BOTHER
#define USE_TERMIOS_SPEED
#endif

struct sp_port {
	char *name;
	char *description;
	enum sp_transport transport;
	int usb_bus;
	int usb_address;
	int usb_vid;
	int usb_pid;
	char *usb_manufacturer;
	char *usb_product;
	char *usb_serial;
	char *bluetooth_address;
#ifdef _WIN32
	char *usb_path;
	HANDLE hdl;
	COMMTIMEOUTS timeouts;
	OVERLAPPED write_ovl;
	OVERLAPPED read_ovl;
	OVERLAPPED wait_ovl;
	DWORD events;
	BYTE *write_buf;
	DWORD write_buf_size;
	BOOL writing;
	BOOL wait_running;
	BOOL last_wait_thread_exited;
#else
	int fd;
#endif
};

struct sp_port_config {
	int baudrate;
	int bits;
	enum sp_parity parity;
	int stopbits;
	enum sp_rts rts;
	enum sp_cts cts;
	enum sp_dtr dtr;
	enum sp_dsr dsr;
	enum sp_xonxoff xon_xoff;
};

struct port_data {
#ifdef _WIN32
	DCB dcb;
#else
	struct termios term;
	int controlbits;
	int termiox_supported;
	int rts_flow;
	int cts_flow;
	int dtr_flow;
	int dsr_flow;
#endif
};

#ifdef _WIN32
typedef HANDLE event_handle;
#else
typedef int event_handle;
#endif

/* Standard baud rates. */
#ifdef _WIN32
#define BAUD_TYPE DWORD
#define BAUD(n) {CBR_##n, n}
#else
#define BAUD_TYPE speed_t
#define BAUD(n) {B##n, n}
#endif

struct std_baudrate {
	BAUD_TYPE index;
	int value;
};

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

static inline bool turbo_serial_tlog_debug_enabled(void) {
	tlog_t *logger = tlog_peek_default();
	return logger && TURBO_LOG_LEVEL_DEBUG >= tlog_get_level(logger);
}

static inline enum sp_return sp_normalize_return(enum sp_return value) {
	switch (value) {
	case SP_OK:
	case SP_ERR_ARG:
	case SP_ERR_FAIL:
	case SP_ERR_MEM:
	case SP_ERR_SUPP:
		return value;
	default:
		return SP_ERR_FAIL;
	}
}

static inline enum sp_return sp_error_return(const char *function, enum sp_return err,
                                             const char *code, const char *reason) {
	TLOG_DEBUG("libserialport error: function={} code={} reason={}", function, code, reason);
	return err;
}

static inline enum sp_return sp_fail_return(const char *function, const char *reason) {
	if (turbo_serial_tlog_debug_enabled()) {
		char *errmsg = sp_last_error_message();
		TLOG_DEBUG("libserialport system failure: function={} reason={} detail={}",
			function, reason, errmsg ? errmsg : "");
		sp_free_error_message(errmsg);
	}
	return SP_ERR_FAIL;
}

struct sp_port **list_append(struct sp_port **list, const char *portname);

/* OS-specific Helper functions. */
enum sp_return get_port_details(struct sp_port *port);
enum sp_return list_ports(struct sp_port ***list);

/* Timing abstraction */

struct time {
	uint64_t ns;
};

struct timeout {
	unsigned int ms, limit_ms;
	struct time start, now, end, delta, delta_max;
	struct timeval delta_tv;
	bool calls_started, overflow;
};

void time_get(struct time *time);
void time_set_ms(struct time *time, unsigned int ms);
void time_add(const struct time *a, const struct time *b, struct time *result);
void time_sub(const struct time *a, const struct time *b, struct time *result);
bool time_greater(const struct time *a, const struct time *b);
void time_as_timeval(const struct time *time, struct timeval *tv);
unsigned int time_as_ms(const struct time *time);
void timeout_start(struct timeout *timeout, unsigned int timeout_ms);
void timeout_limit(struct timeout *timeout, unsigned int limit_ms);
bool timeout_check(struct timeout *timeout);
void timeout_update(struct timeout *timeout);
struct timeval *timeout_timeval(struct timeout *timeout);
unsigned int timeout_remaining_ms(struct timeout *timeout);


#endif /* TURBOSERIAL_API_ONLY */

#ifdef __cplusplus
}
#endif


#endif /* TURBOSERIAL_SERIALPORT_H */
