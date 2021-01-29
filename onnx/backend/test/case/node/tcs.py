# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from onnx import TensorProto
from ..base import Base
from . import expect


class TCS(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node('TCS',
            inputs=['A','B'],
            outputs=['Y'],
        )

        # expected scale 0.0196078438 and zero point 153
        A = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float32)
        B = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float32)
        Y = (2*A)+B

        expect(node, inputs=[A,B], outputs=[Y],
               name='test_tcs')
