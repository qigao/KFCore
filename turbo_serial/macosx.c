#include <config.h>
#include "serialport.h"

enum sp_return get_port_details(struct sp_port *port)
{
	/*
	 * Description limited to 127 char, anything longer
	 * would not be user friendly anyway.
	 */
	char description[128];
	int bus, address, vid, pid = -1;
	char manufacturer[128], product[128], serial[128];
	CFMutableDictionaryRef classes;
	io_iterator_t iter;
	io_object_t ioport, ioparent;
	CFTypeRef cf_property, cf_bus, cf_address, cf_vendor, cf_product;
	Boolean result;
	char path[PATH_MAX], class[64];

	TLOG_DEBUG("Getting serial port list");
	if (!(classes = IOServiceMatching(kIOSerialBSDServiceValue)))
		return sp_fail_return(__func__, "IOServiceMatching() failed");

	if (IOServiceGetMatchingServices(kIOMainPortDefault, classes,
	                                 &iter) != KERN_SUCCESS)
		return sp_fail_return(__func__, "IOServiceGetMatchingServices() failed");

	TLOG_DEBUG("Iterating over results");
	while ((ioport = IOIteratorNext(iter))) {
		if (!(cf_property = IORegistryEntryCreateCFProperty(ioport,
		            CFSTR(kIOCalloutDeviceKey), kCFAllocatorDefault, 0))) {
			IOObjectRelease(ioport);
			continue;
		}
		result = CFStringGetCString(cf_property, path, sizeof(path),
		                            kCFStringEncodingASCII);
		CFRelease(cf_property);
		if (!result || strcmp(path, port->name)) {
			IOObjectRelease(ioport);
			continue;
		}
		TLOG_DEBUG("Found port {}", path);

		IORegistryEntryGetParentEntry(ioport, kIOServicePlane, &ioparent);
		if ((cf_property=IORegistryEntrySearchCFProperty(ioparent,kIOServicePlane,
		           CFSTR("IOClass"), kCFAllocatorDefault,
		           kIORegistryIterateRecursively | kIORegistryIterateParents))) {
			if (CFStringGetCString(cf_property, class, sizeof(class),
			                       kCFStringEncodingASCII) &&
			    strstr(class, "USB")) {
				TLOG_DEBUG("Found USB class device");
				port->transport = SP_TRANSPORT_USB;
			}
			CFRelease(cf_property);
		}
		if ((cf_property=IORegistryEntrySearchCFProperty(ioparent,kIOServicePlane,
		           CFSTR("IOProviderClass"), kCFAllocatorDefault,
		           kIORegistryIterateRecursively | kIORegistryIterateParents))) {
			if (CFStringGetCString(cf_property, class, sizeof(class),
			                       kCFStringEncodingASCII) &&
			    strstr(class, "USB")) {
				TLOG_DEBUG("Found USB class device");
				port->transport = SP_TRANSPORT_USB;
			}
			CFRelease(cf_property);
		}
		IOObjectRelease(ioparent);

		if ((cf_property = IORegistryEntrySearchCFProperty(ioport,kIOServicePlane,
		         CFSTR("USB Interface Name"), kCFAllocatorDefault,
		         kIORegistryIterateRecursively | kIORegistryIterateParents)) ||
		    (cf_property = IORegistryEntrySearchCFProperty(ioport,kIOServicePlane,
		         CFSTR("USB Product Name"), kCFAllocatorDefault,
		         kIORegistryIterateRecursively | kIORegistryIterateParents)) ||
		    (cf_property = IORegistryEntrySearchCFProperty(ioport,kIOServicePlane,
		         CFSTR("Product Name"), kCFAllocatorDefault,
		         kIORegistryIterateRecursively | kIORegistryIterateParents)) ||
		    (cf_property = IORegistryEntryCreateCFProperty(ioport,
		         CFSTR(kIOTTYDeviceKey), kCFAllocatorDefault, 0))) {
			if (CFStringGetCString(cf_property, description, sizeof(description),
			                       kCFStringEncodingASCII)) {
				TLOG_DEBUG("Found description {}", description);
				port->description = strdup(description);
			}
			CFRelease(cf_property);
		} else {
			TLOG_DEBUG("No description for this device");
		}

		cf_bus = IORegistryEntrySearchCFProperty(ioport, kIOServicePlane,
		                                         CFSTR("USBBusNumber"),
		                                         kCFAllocatorDefault,
		                                         kIORegistryIterateRecursively
		                                         | kIORegistryIterateParents);
		cf_address = IORegistryEntrySearchCFProperty(ioport, kIOServicePlane,
		                                         CFSTR("USB Address"),
		                                         kCFAllocatorDefault,
		                                         kIORegistryIterateRecursively
		                                         | kIORegistryIterateParents);
		if (cf_bus && cf_address &&
		    CFNumberGetValue(cf_bus    , kCFNumberIntType, &bus) &&
		    CFNumberGetValue(cf_address, kCFNumberIntType, &address)) {
			TLOG_DEBUG("Found matching USB bus:address {}:{}", bus, address);
			port->usb_bus = bus;
			port->usb_address = address;
		}
		if (cf_bus)
			CFRelease(cf_bus);
		if (cf_address)
			CFRelease(cf_address);

		cf_vendor = IORegistryEntrySearchCFProperty(ioport, kIOServicePlane,
		                                         CFSTR("idVendor"),
		                                         kCFAllocatorDefault,
		                                         kIORegistryIterateRecursively
		                                         | kIORegistryIterateParents);
		cf_product = IORegistryEntrySearchCFProperty(ioport, kIOServicePlane,
		                                         CFSTR("idProduct"),
		                                         kCFAllocatorDefault,
		                                         kIORegistryIterateRecursively
		                                         | kIORegistryIterateParents);
		if (cf_vendor && cf_product &&
		    CFNumberGetValue(cf_vendor , kCFNumberIntType, &vid) &&
		    CFNumberGetValue(cf_product, kCFNumberIntType, &pid)) {
			TLOG_DEBUG("Found matching USB VID:PID {}:{}", vid, pid);
			port->usb_vid = vid;
			port->usb_pid = pid;
		}
		if (cf_vendor)
			CFRelease(cf_vendor);
		if (cf_product)
			CFRelease(cf_product);

		if ((cf_property = IORegistryEntrySearchCFProperty(ioport,kIOServicePlane,
		         CFSTR("USB Vendor Name"), kCFAllocatorDefault,
		         kIORegistryIterateRecursively | kIORegistryIterateParents))) {
			if (CFStringGetCString(cf_property, manufacturer, sizeof(manufacturer),
			                       kCFStringEncodingASCII)) {
				TLOG_DEBUG("Found manufacturer {}", manufacturer);
				port->usb_manufacturer = strdup(manufacturer);
			}
			CFRelease(cf_property);
		}

		if ((cf_property = IORegistryEntrySearchCFProperty(ioport,kIOServicePlane,
		         CFSTR("USB Product Name"), kCFAllocatorDefault,
		         kIORegistryIterateRecursively | kIORegistryIterateParents))) {
			if (CFStringGetCString(cf_property, product, sizeof(product),
			                       kCFStringEncodingASCII)) {
				TLOG_DEBUG("Found product name {}", product);
				port->usb_product = strdup(product);
			}
			CFRelease(cf_property);
		}

		if ((cf_property = IORegistryEntrySearchCFProperty(ioport,kIOServicePlane,
		         CFSTR("USB Serial Number"), kCFAllocatorDefault,
		         kIORegistryIterateRecursively | kIORegistryIterateParents))) {
			if (CFStringGetCString(cf_property, serial, sizeof(serial),
			                       kCFStringEncodingASCII)) {
				TLOG_DEBUG("Found serial number {}", serial);
				port->usb_serial = strdup(serial);
			}
			CFRelease(cf_property);
		}

		IOObjectRelease(ioport);
		break;
	}
	IOObjectRelease(iter);

	return SP_OK;
}

enum sp_return list_ports(struct sp_port ***list)
{
	CFMutableDictionaryRef classes;
	io_iterator_t iter;
	char path[PATH_MAX];
	io_object_t port;
	CFTypeRef cf_path;
	Boolean result;
	int ret = SP_OK;

	TLOG_DEBUG("Creating matching dictionary");
	if (!(classes = IOServiceMatching(kIOSerialBSDServiceValue))) {
		ret = sp_fail_return(__func__, "IOServiceMatching() failed");
		goto out_done;
	}

	TLOG_DEBUG("Getting matching services");
	if (IOServiceGetMatchingServices(kIOMainPortDefault, classes,
	                                 &iter) != KERN_SUCCESS) {
		ret = sp_fail_return(__func__, "IOServiceGetMatchingServices() failed");
		goto out_done;
	}

	TLOG_DEBUG("Iterating over results");
	while ((port = IOIteratorNext(iter))) {
		cf_path = IORegistryEntryCreateCFProperty(port,
				CFSTR(kIOCalloutDeviceKey), kCFAllocatorDefault, 0);
		if (cf_path) {
			result = CFStringGetCString(cf_path, path, sizeof(path),
			                            kCFStringEncodingASCII);
			CFRelease(cf_path);
			if (result) {
				TLOG_DEBUG("Found port {}", path);
				if (!(*list = list_append(*list, path))) {
					ret = sp_error_return(__func__, SP_ERR_MEM, "SP_ERR_MEM", "List append failed");
					IOObjectRelease(port);
					goto out;
				}
			}
		}
		IOObjectRelease(port);
	}
out:
	IOObjectRelease(iter);
out_done:

	return ret;
}
