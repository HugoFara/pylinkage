#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The exceptions module is a simple quick way to access the built-in exceptions.

Created on Wed Jun 16, 15:20:06 2021.

@author: HugoFara
"""


class UnbuildableError(Exception):
    """Should be raised when the constraints cannot be solved."""

    def __init__(self, joint, message='Unable to solve constraints'):
        self.joint = joint
        self.message = message
        super().__init__(message)

    def __str__(self):
        """
        Output the problematic joint.

        Returns
        -------
        str
            Name of the joint that can't be solved.

        """
        return f"{self.message} on {self.joint}"


class HypostaticError(Exception):
    """The system is under-constrained and multiple solutions may exist."""

    def __init__(self, linkage, message='The system is hypo-static!'):
        self.linkage = linkage
        super().__init__(message)
