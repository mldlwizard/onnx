# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class TestOp(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'TestOp',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = 2 * x  # expected output [4., 10., 18.]
        expect(node, inputs=[x], outputs=[y],
               name='test_operator_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = 2 * x
        expect(node, inputs=[x], outputs=[y],
               name='test_mul')
