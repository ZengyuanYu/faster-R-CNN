#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-3-2
import numpy as np
import numpy as xp

import six
from six import __init__

def loc2bbox(src_bbox, loc):
	if src_bbox.shape[0] == 0:
		return xp.zeros((0, 4), dtype=loc.dtype)
	
	src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
	
	src_height = src_bbox[:, 2] - src_bbox[:, 0]
	src_width = src_bbox[:, 3] - src_bbox[:, 1]
	src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
	src_ctr_x = src_bbox[:, 1] + 0.5 * src_width
	
	dy = loc[:, 0::4]
	dx = loc[:, 1::4]
	dh = loc[:, 2::4]
	dw = loc[:, 3::4]
	
	ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis]
	ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]
	h = xp.exp(dh) * src_height[:, xp.newaxis]
	w = xp.exp(dw) * src_width[:, xp.newaxis]
	
	dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
	dst_bbox[:, 0::4] = ctr_y - 0.5 * h
	dst_bbox[:, 1::4] = ctr_x - 0.5 * w
	dst_bbox[:, 2::4] = ctr_y + 0.5 * h
	dst_bbox[:, 3::4] = ctr_x + 0.5 * w
	
	return dst_bbox





















