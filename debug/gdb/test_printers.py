#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the entry iterator behind the gdb pretty-printers in printers.py.

printers.py imports gdb, which exists only inside a gdb process, so a stub
module stands in for it here.

Usage: python3 debug/gdb/test_printers.py
"""

import os
import sys
import types

sys.modules["gdb"] = types.ModuleType("gdb")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import printers

FAILURES = []


def entries(rows, cols, row_major):
    return list(printers._MatrixEntryIterator(rows, cols, row_major))


def check(condition, message):
    if not condition:
        FAILURES.append(message)
        print("FAIL: %s" % message)


# Entries come out in storage order.
check(entries(2, 3, 0) == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)], "2x3 column-major order")
check(entries(2, 3, 1) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], "2x3 row-major order")
check(entries(1, 1, 0) == [(0, 0)], "1x1")

# An empty matrix or block has a null data pointer, so no entry may be yielded when either dimension is 0, whatever the
# storage order and whichever dimension is the outer one.
for row_major in (0, 1):
    for rows, cols in ((0, 3), (3, 0), (0, 1), (1, 0), (0, 0)):
        check(entries(rows, cols, row_major) == [],
              "%dx%d %s yields no entries" % (rows, cols, "row-major" if row_major else "column-major"))

if FAILURES:
    print("%d failure(s)" % len(FAILURES))
    sys.exit(1)
print("OK")
