# -*- coding: utf-8 -*-
# This file is part of Eigen, a lightweight C++ template library
# for linear algebra.
#
# Copyright (C) 2009 Benjamin Schindler <bschindler@inf.ethz.ch>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

# Pretty printers for Eigen::Matrix
# This is still pretty basic as the python extension to gdb is still pretty basic. 
# It cannot handle complex eigen types and it doesn't support many of the other eigen types
# This code supports fixed size as well as dynamic size matrices

# To use it:
#
# * Create a directory and put the file as well as an empty __init__.py in 
#   that directory.
# * Create a ~/.gdbinit file, that contains the following:
#      python
#      import sys
#      sys.path.insert(0, '/path/to/eigen/printer/directory')
#      from printers import register_eigen_printers
#      register_eigen_printers(None)
#      end

import gdb
import re
from bisect import bisect_left


# Basic row/column iteration code for use with Sparse and Dense matrices
class _MatrixEntryIterator(object):
	
	def __init__(self, rows, cols, row_major):
		self.rows = rows
		self.cols = cols
		self.currentRow = 0
		self.currentCol = 0
		self.rowMajor = row_major

	def __iter__(self):
		return self

	def next(self):
		return self.__next__()  # Python 2.x compatibility

	def __next__(self):
		row = self.currentRow
		col = self.currentCol
		# An empty matrix has a null data pointer, so stop when either dimension is 0.
		if row >= self.rows or col >= self.cols:
			raise StopIteration

		if self.rowMajor == 0:
			self.currentRow += 1
			if self.currentRow >= self.rows:
				self.currentRow = 0
				self.currentCol += 1
		else:
			self.currentCol += 1
			if self.currentCol >= self.cols:
				self.currentCol = 0
				self.currentRow += 1

		return row, col


class EigenMatrixPrinter:
	"""Print Eigen Matrix or Array of some kind"""

	def __init__(self, variety, val):
		"""Extract all the necessary information"""
		
		# Save the variety (presumably "Matrix" or "Array") for later usage
		self.variety = variety
		
		# The gdb extension does not support value template arguments - need to extract them by hand
		typeinfo = val.type
		if typeinfo.code == gdb.TYPE_CODE_REF:
			typeinfo = typeinfo.target()
		self.type = typeinfo.unqualified().strip_typedefs()
		tag = self.type.tag
		regex = re.compile('<.*>')
		m = regex.findall(tag)[0][1:-1]
		template_params = m.split(',')
		template_params = [x.replace(" ", "") for x in template_params]
		
		if template_params[1] in ['-0x00000000000000001', '-0x000000001', '-1']:
			self.rows = val['m_storage']['m_rows']
		else:
			self.rows = int(template_params[1])
		
		if template_params[2] in ['-0x00000000000000001', '-0x000000001', '-1']:
			self.cols = val['m_storage']['m_cols']
		else:
			self.cols = int(template_params[2])
		
		self.options = 0  # default value
		if len(template_params) > 3:
			self.options = template_params[3]
		
		self.rowMajor = (int(self.options) & 0x1)
		
		self.innerType = self.type.template_argument(0)
		
		self.val = val
		
		# Fixed size matrices have a struct as their storage, so we need to walk through this
		self.data = self.val['m_storage']['m_data']
		if self.data.type.code == gdb.TYPE_CODE_STRUCT:
			self.data = self.data['array']
			self.data = self.data.cast(self.innerType.pointer())
			
	class _Iterator(_MatrixEntryIterator):
		def __init__(self, rows, cols, data_ptr, row_major):
			super(EigenMatrixPrinter._Iterator, self).__init__(rows, cols, row_major)

			self.dataPtr = data_ptr

		def __next__(self):
			row, col = super(EigenMatrixPrinter._Iterator, self).__next__()
			
			item = self.dataPtr.dereference()
			self.dataPtr += 1
			if self.cols == 1:  # if it's a column vector
				return '[%d]' % (row,), item
			elif self.rows == 1:  # if it's a row vector
				return '[%d]' % (col,), item
			return '[%d,%d]' % (row, col), item
			
	def children(self):
		return self._Iterator(self.rows, self.cols, self.data, self.rowMajor)
		
	def to_string(self):
		return "Eigen::%s<%s,%d,%d,%s> (data ptr: %s)" % (
			self.variety, self.innerType, self.rows, self.cols,
			"RowMajor" if self.rowMajor else "ColMajor", self.data)


class EigenSparseMatrixPrinter:
	"""Print an Eigen SparseMatrix"""

	def __init__(self, val):
		"""Extract all the necessary information"""

		typeinfo = val.type
		if typeinfo.code == gdb.TYPE_CODE_REF:
			typeinfo = typeinfo.target()
		self.type = typeinfo.unqualified().strip_typedefs()
		tag = self.type.tag
		regex = re.compile('<.*>')
		m = regex.findall(tag)[0][1:-1]
		template_params = m.split(',')
		template_params = [x.replace(" ", "") for x in template_params]

		self.options = 0
		if len(template_params) > 1:
			self.options = template_params[1]
		
		self.rowMajor = (int(self.options) & 0x1)
		
		self.innerType = self.type.template_argument(0)
		
		self.val = val

		self.data = self.val['m_data']
		self.data = self.data.cast(self.innerType.pointer())

	class _Iterator(_MatrixEntryIterator):
		def __init__(self, rows, cols, val, row_major):
			super(EigenSparseMatrixPrinter._Iterator, self).__init__(rows, cols, row_major)

			self.val = val
			
		def __next__(self):
			row, col = super(EigenSparseMatrixPrinter._Iterator, self).__next__()
				
			# repeat calculations from SparseMatrix.h:
			outer = row if self.rowMajor else col
			inner = col if self.rowMajor else row
			start = self.val['m_outerIndex'][outer]
			end = (
				(start + self.val['m_innerNonZeros'][outer])
				if self.val['m_innerNonZeros'] else self.val['m_outerIndex'][outer+1]
			)

			# and from CompressedStorage.h:
			data = self.val['m_data']
			if start >= end:
				item = 0
			elif (end > start) and (inner == data['m_indices'][end-1]):
				item = data['m_values'][end-1]
			else:
				# create Python index list from the target range within m_indices
				indices = [data['m_indices'][x] for x in range(int(start), int(end)-1)]
				# find the index with binary search
				idx = int(start) + bisect_left(indices, inner)
				if idx < end and data['m_indices'][idx] == inner:
					item = data['m_values'][idx]
				else:
					item = 0

			return '[%d,%d]' % (row, col), item

	def children(self):
		if self.data:
			return self._Iterator(self.rows(), self.cols(), self.val, self.rowMajor)

		return iter([])   # empty matrix, for now

	def rows(self):
		return self.val['m_outerSize'] if self.rowMajor else self.val['m_innerSize']

	def cols(self):
		return self.val['m_innerSize'] if self.rowMajor else self.val['m_outerSize']

	def to_string(self):

		if self.data:
			status = ("not compressed" if self.val['m_innerNonZeros'] else "compressed")
		else:
			status = "empty"
		dimensions = "%d x %d" % (self.rows(), self.cols())
		layout = "row" if self.rowMajor else "column"

		return "Eigen::SparseMatrix<%s>, %s, %s major, %s" % (
			self.innerType, dimensions, layout, status)


class EigenQuaternionPrinter:
	"""Print an Eigen Quaternion"""
	
	def __init__(self, val):
		"""Extract all the necessary information"""
		# The gdb extension does not support value template arguments - need to extract them by hand
		typeinfo = val.type
		if typeinfo.code == gdb.TYPE_CODE_REF:
			typeinfo = typeinfo.target()
		self.type = typeinfo.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		
		# Quaternions have a struct as their storage, so we need to walk through this
		self.data = self.val['m_coeffs']['m_storage']['m_data']['array']
		self.data = self.data.cast(self.innerType.pointer())
			
	class _Iterator:
		def __init__(self, data_ptr):
			self.dataPtr = data_ptr
			self.currentElement = 0
			self.elementNames = ['x', 'y', 'z', 'w']
			
		def __iter__(self):
			return self
	
		def next(self):
			return self.__next__()  # Python 2.x compatibility

		def __next__(self):
			element = self.currentElement
			
			if self.currentElement >= 4:  # there are 4 elements in a quaternion
				raise StopIteration
			
			self.currentElement += 1
			
			item = self.dataPtr.dereference()
			self.dataPtr += 1
			return '[%s]' % (self.elementNames[element],), item
			
	def children(self):
		return self._Iterator(self.data)
	
	def to_string(self):
		return "Eigen::Quaternion<%s> (data ptr: %s)" % (self.innerType, self.data)


_DYNAMIC_SPELLINGS = ['-0x00000000000000001', '-0x000000001', '-1']


def _split_template_arguments(tag):
	"""Split the top-level template arguments of 'Name<a, B<c, d>, e>' into ['a', 'B<c, d>', 'e']."""
	arguments = []
	depth = 0
	current = ''
	for char in tag[tag.index('<') + 1:tag.rindex('>')]:
		if char == '<':
			depth += 1
		elif char == '>':
			depth -= 1
		if char == ',' and depth == 0:
			arguments.append(current.strip())
			current = ''
		else:
			current += char
	arguments.append(current.strip())
	return arguments


class EigenBlockPrinter:
	"""Print an Eigen Block or VectorBlock of a plain Matrix or Array.

	The block is read through its own data pointer, sizes and outer stride, so that blocks that are not
	contiguous (a sub-block, a row of a column-major matrix) show the right coefficients."""

	def __init__(self, val):
		if val.type.code == gdb.TYPE_CODE_REF:
			val = val.referenced_value()
		typeinfo = val.type.unqualified().strip_typedefs()
		if typeinfo.tag.startswith('Eigen::VectorBlock<'):
			# A VectorBlock adds nothing to the Block it derives from.
			typeinfo = typeinfo.fields()[0].type
			val = val.cast(typeinfo)
		self.type = typeinfo.unqualified().strip_typedefs()

		arguments = _split_template_arguments(self.type.tag)
		if not re.match(r'^(const )?Eigen::(Matrix|Array)<', arguments[0]):
			raise ValueError('only blocks of a plain Matrix or Array are supported')
		# Dynamic is -1, which some gdb versions print in hexadecimal (see EigenMatrixPrinter).
		rows_at_compile_time = -1 if arguments[1].replace(" ", "") in _DYNAMIC_SPELLINGS else int(arguments[1])
		cols_at_compile_time = -1 if arguments[2].replace(" ", "") in _DYNAMIC_SPELLINGS else int(arguments[2])

		parent = val['m_xpr']
		if parent.type.strip_typedefs().code in (gdb.TYPE_CODE_REF, getattr(gdb, 'TYPE_CODE_RVALUE_REF', None)):
			parent = parent.referenced_value()
		parent = EigenMatrixPrinter("Matrix", parent)
		self.innerType = parent.innerType
		self.rows = int(val['m_rows']['m_value']) if rows_at_compile_time == -1 else rows_at_compile_time
		self.cols = int(val['m_cols']['m_value']) if cols_at_compile_time == -1 else cols_at_compile_time

		# Block's storage order, as in internal::traits<Block>
		if rows_at_compile_time == 1 and cols_at_compile_time != 1:
			self.rowMajor = 1
		elif cols_at_compile_time == 1 and rows_at_compile_time != 1:
			self.rowMajor = 0
		else:
			self.rowMajor = parent.rowMajor

		# A plain parent has inner stride 1 and outer stride rows (column-major) or cols (row-major);
		# see BlockImpl_dense::innerStride() and outer_stride().
		outer_stride = int(val['m_outerStride'])
		parent_outer_stride = int(parent.cols if parent.rowMajor else parent.rows)
		inner_stride = 1 if self.rowMajor == parent.rowMajor else parent_outer_stride
		self.rowStride = outer_stride if self.rowMajor else inner_stride
		self.colStride = inner_stride if self.rowMajor else outer_stride

		self.data = val['m_data']
		if self.data.type.strip_typedefs().code != gdb.TYPE_CODE_PTR:
			raise ValueError('unexpected block data member')

	class _Iterator(_MatrixEntryIterator):
		def __init__(self, printer):
			super(EigenBlockPrinter._Iterator, self).__init__(printer.rows, printer.cols, printer.rowMajor)
			self.printer = printer

		def __next__(self):
			row, col = super(EigenBlockPrinter._Iterator, self).__next__()

			item = (self.printer.data + (row * self.printer.rowStride + col * self.printer.colStride)).dereference()
			if self.cols == 1:  # if it's a column vector
				return '[%d]' % (row,), item
			elif self.rows == 1:  # if it's a row vector
				return '[%d]' % (col,), item
			return '[%d,%d]' % (row, col), item

	def children(self):
		return self._Iterator(self)

	def to_string(self):
		return "Eigen::Block<%s,%d,%d,%s> (data ptr: %s, row stride: %d, column stride: %d)" % (
			self.innerType, self.rows, self.cols, "RowMajor" if self.rowMajor else "ColMajor",
			self.data, self.rowStride, self.colStride)


def make_eigen_block_printer(val):
	"""Return a printer for a Block or VectorBlock, or None to leave it to gdb's raw struct printing.

	Blocks of anything but a plain Matrix or Array, and blocks whose layout cannot be read, are declined
	rather than shown with wrong coefficients."""
	try:
		return EigenBlockPrinter(val)
	except (gdb.error, RuntimeError, ValueError, KeyError, IndexError, TypeError):
		return None


def build_eigen_dictionary():
	pretty_printers_dict[re.compile('^Eigen::Quaternion<.*>$')] = lambda val: EigenQuaternionPrinter(val)
	pretty_printers_dict[re.compile('^Eigen::Matrix<.*>$')] = lambda val: EigenMatrixPrinter("Matrix", val)
	pretty_printers_dict[re.compile('^Eigen::Block<.*>$')] = make_eigen_block_printer
	pretty_printers_dict[re.compile('^Eigen::VectorBlock<.*>$')] = make_eigen_block_printer
	pretty_printers_dict[re.compile('^Eigen::SparseMatrix<.*>$')] = lambda val: EigenSparseMatrixPrinter(val)
	pretty_printers_dict[re.compile('^Eigen::Array<.*>$')] = lambda val: EigenMatrixPrinter("Array",  val)


def register_eigen_printers(obj):
	"""Register eigen pretty-printers with objfile Obj"""

	if obj is None:
		obj = gdb
	obj.pretty_printers.append(lookup_function)


def lookup_function(val):
	"""Look-up and return a pretty-printer that can print val."""
	
	typeinfo = val.type
	
	if typeinfo.code == gdb.TYPE_CODE_REF:
		typeinfo = typeinfo.target()
	
	typeinfo = typeinfo.unqualified().strip_typedefs()
	
	typename = typeinfo.tag
	if typename is None:
		return None
	
	for function in pretty_printers_dict:
		if function.search(typename):
			return pretty_printers_dict[function](val)
	
	return None


pretty_printers_dict = {}

build_eigen_dictionary()
