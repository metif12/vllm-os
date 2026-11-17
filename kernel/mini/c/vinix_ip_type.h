/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Accessors for the `type` field of lwIP's struct ip_addr.
 * V's `@type` escape maps to `_v_type` in generated C, which does not
 * exist on real C structs (see netcore/core.v). Call these instead. */
#ifndef VINIX_IP_TYPE_H
#define VINIX_IP_TYPE_H

#include <stdint.h>

uint8_t vinix_ip_get_type(const void *addr);
void vinix_ip_set_type(void *addr, uint8_t t);

#endif
