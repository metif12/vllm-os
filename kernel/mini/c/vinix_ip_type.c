/* SPDX-License-Identifier: GPL-2.0-or-later */
#include "vinix_ip_type.h"
#include <lwip/ip_addr.h>

uint8_t vinix_ip_get_type(const void *addr) {
	return IP_GET_TYPE((const ip_addr_t *)addr);
}

void vinix_ip_set_type(void *addr, uint8_t t) {
	IP_SET_TYPE((ip_addr_t *)addr, t);
}
