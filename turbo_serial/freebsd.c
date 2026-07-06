#include <config.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <dirent.h>
#include <libusb20.h>
#include <libusb20_desc.h>
#include "serialport.h"

#define DEV_CUA_PATH "/dev/cua"

static char *strrspn(const char *s, const char *charset)
{
	char *t = (char *)s + strlen(s);
	while (t != (char *)s)
		if (!strchr(charset, *(--t)))
			return ++t;
	return t;
}

static int strend(const char *str, const char *pattern)
{
	size_t slen = strlen(str);
	size_t plen = strlen(pattern);
	if (slen >= plen)
		return (!memcmp(pattern, (str + slen - plen), plen));
	return 0;
}

static int libusb_query_port(struct libusb20_device *dev, int idx,
			     char **drv_name_str, char **drv_inst_str)
{
	int rc;
	char *j;
	char sbuf[FILENAME_MAX];

	if (!drv_name_str || !drv_inst_str)
		return -1;

	rc = libusb20_dev_kernel_driver_active(dev, idx);
	if (rc < 0)
		return rc;

	sbuf[0] = 0;
	libusb20_dev_get_iface_desc(dev, idx, (char *)&sbuf,
				    (uint8_t)sizeof(sbuf) - 1);
	if (sbuf[0] == 0)
		return rc;

	j = strchr(sbuf, ':');
	if (j > sbuf) {
		sbuf[j - sbuf] = 0;
		/*
		 * The device driver name may contain digits that
		 * is not a part of the device instance number - like "u3g".
		 */
		j = strrspn(sbuf, "0123456789");
		if (j > sbuf) {
			*drv_name_str = strndup(sbuf, j - sbuf);
			*drv_inst_str = strdup((j));
		}
	}

	return rc;
}

static int sysctl_query_dev_drv(const char *drv_name_str,
	const char *drv_inst_str, char **ttyname, int *const ttyport_cnt)
{
	int rc;
	char sbuf[FILENAME_MAX];
	char tbuf[FILENAME_MAX];
	size_t tbuf_len;

	if (!ttyname || !ttyport_cnt)
		return -1;

	snprintf(sbuf, sizeof(sbuf), "dev.%s.%s.ttyname", drv_name_str,
		 drv_inst_str);
	tbuf_len = sizeof(tbuf) - 1;
	if ((rc = sysctlbyname(sbuf, tbuf, &tbuf_len, NULL, 0)) != 0)
		return rc;

	tbuf[tbuf_len] = 0;
	*ttyname = strndup(tbuf, tbuf_len);
	snprintf(sbuf, sizeof(sbuf), "dev.%s.%s.ttyports",
		 drv_name_str, drv_inst_str);
	tbuf_len = sizeof(tbuf) - 1;
	rc = sysctlbyname(sbuf, tbuf, &tbuf_len, NULL, 0);
	if (rc == 0) {
		*ttyport_cnt = *(uint32_t *)tbuf;
	} else {
		*ttyport_cnt = 0;
	}

	return rc;
}

static int populate_port_struct_from_libusb_desc(struct sp_port *const port,
						 struct libusb20_device *dev)
{
	char tbuf[FILENAME_MAX];

	/* Populate port structure from libusb description. */
	struct LIBUSB20_DEVICE_DESC_DECODED *dev_desc =
		libusb20_dev_get_device_desc(dev);

	if (!dev_desc)
		return -1;

	port->transport = SP_TRANSPORT_USB;
	port->usb_vid = dev_desc->idVendor;
	port->usb_pid = dev_desc->idProduct;
	port->usb_bus = libusb20_dev_get_bus_number(dev);
	port->usb_address = libusb20_dev_get_address(dev);
	if (libusb20_dev_req_string_simple_sync
	    (dev, dev_desc->iManufacturer, tbuf, sizeof(tbuf)) == 0) {
		port->usb_manufacturer = strdup(tbuf);
	}
	if (libusb20_dev_req_string_simple_sync
	    (dev, dev_desc->iProduct, tbuf, sizeof(tbuf)) == 0) {
		port->usb_product = strdup(tbuf);
	}
	if (libusb20_dev_req_string_simple_sync
	    (dev, dev_desc->iSerialNumber, tbuf, sizeof(tbuf)) == 0) {
		port->usb_serial = strdup(tbuf);
	}
	/* If present, add serial to description for better identification. */
	tbuf[0] = '\0';
	if (port->usb_product && port->usb_product[0])
		strncat(tbuf, port->usb_product, sizeof(tbuf) - 1);
	else
		strncat(tbuf, libusb20_dev_get_desc(dev), sizeof(tbuf) - 1);
	if (port->usb_serial && port->usb_serial[0]) {
		strncat(tbuf, " ", sizeof(tbuf) - 1);
		strncat(tbuf, port->usb_serial, sizeof(tbuf) - 1);
	}
	port->description = strdup(tbuf);
	port->bluetooth_address = NULL;

	return 0;
}

enum sp_return get_port_details(struct sp_port *port)
{
	int rc;
	struct libusb20_backend *be;
	struct libusb20_device *dev, *dev_last;
	char tbuf[FILENAME_MAX];
	char *cua_sfx;
	int cua_dev_found;
	uint8_t idx;
	int sub_inst;

	if (!strncmp(port->name, DEV_CUA_PATH, strlen(DEV_CUA_PATH))) {
		cua_sfx = port->name + strlen(DEV_CUA_PATH);
	} else {
		return sp_error_return(__func__, SP_ERR_ARG, "SP_ERR_ARG", "Device name not recognized");
	}

	/* Native UART enumeration. */
	if ((cua_sfx[0] == 'u') || (cua_sfx[0] == 'd')) {
		port->transport = SP_TRANSPORT_NATIVE;
		snprintf(tbuf, sizeof(tbuf), "cua%s", cua_sfx);
		port->description = strdup(tbuf);
		return SP_OK;
	}

	/* USB device enumeration. */
	dev = dev_last = NULL;
	be = libusb20_be_alloc_default();
	cua_dev_found = 0;
	while (cua_dev_found == 0) {
		dev = libusb20_be_device_foreach(be, dev_last);
		if (!dev)
			break;

		libusb20_dev_open(dev, 0);

		for (idx = 0; idx <= UINT8_MAX - 1; idx++) {
			char *drv_name_str = NULL;
			char *drv_inst_str = NULL;
			char *ttyname = NULL;
			int ttyport_cnt;

			rc = libusb_query_port(dev, idx, &drv_name_str, &drv_inst_str);
			if (rc == 0) {
				rc = sysctl_query_dev_drv(drv_name_str,
					  drv_inst_str, &ttyname, &ttyport_cnt);
				if (rc == 0) {
					/* Handle multiple subinstances of serial ports in the same driver instance. */
					for (sub_inst = 0; sub_inst < ttyport_cnt; sub_inst++) {
						if (ttyport_cnt == 1)
							snprintf(tbuf, sizeof(tbuf), "%s", ttyname);
						else
							snprintf(tbuf, sizeof(tbuf), "%s.%d", ttyname, sub_inst);
						if (!strcmp(cua_sfx, tbuf)) {
							cua_dev_found = 1;
							populate_port_struct_from_libusb_desc(port, dev);
							break; /* Break out of sub instance loop. */
						}
					}
				}
			}

			/* Clean up. */
			if (ttyname)
				free(ttyname);
			if (drv_name_str)
				free(drv_name_str);
			if (drv_inst_str)
				free(drv_inst_str);
			if (cua_dev_found)
				break; /* Break out of USB device port idx loop. */
		}
		libusb20_dev_close(dev);
		dev_last = dev;
	}
	libusb20_be_free(be);

	return SP_OK;
}

enum sp_return list_ports(struct sp_port ***list)
{
	DIR *dir;
	struct dirent *entry;
	struct termios tios;
	char name[PATH_MAX];
	int fd, ret;

	TLOG_DEBUG("Enumerating tty devices");
	if (!(dir = opendir("/dev")))
		return sp_fail_return(__func__, "Could not open dir /dev");

	TLOG_DEBUG("Iterating over results");
	while ((entry = readdir(dir))) {
		ret = SP_OK;
		if (entry->d_type != DT_CHR)
			continue;
		if (strncmp(entry->d_name, "cuaU", 4) != 0)
			if (strncmp(entry->d_name, "cuau", 4) != 0)
				if (strncmp(entry->d_name, "cuad", 4) != 0)
					continue;
		if (strend(entry->d_name, ".init"))
			continue;
		if (strend(entry->d_name, ".lock"))
			continue;

		snprintf(name, sizeof(name), "/dev/%s", entry->d_name);
		TLOG_DEBUG("Found device {}", name);

		/* Check that we can open tty/cua device in rw mode - we need that. */
		if ((fd = open(name, O_RDWR | O_NONBLOCK | O_NOCTTY | O_TTY_INIT | O_CLOEXEC)) < 0) {
			TLOG_DEBUG("Open failed, skipping");
			continue;
		}

		/* Sanity check if we got a real tty. */
		if (!isatty(fd)) {
			close(fd);
			continue;
		}

		ret = tcgetattr(fd, &tios);
		close(fd);
		if (ret < 0 || cfgetospeed(&tios) <= 0 || cfgetispeed(&tios) <= 0)
			continue;

		TLOG_DEBUG("Found port {}", name);

		*list = list_append(*list, name);
		if (!*list) {
			ret = sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "List append failed");
			break;
		}
	}
	closedir(dir);

	return ret;
}
