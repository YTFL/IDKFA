"""Imitation Learning & Behavioral Cloning pipeline (training-only, never imported in submission/)."""

from .teacher import PrivilegedTeacherOracle

__all__ = ["PrivilegedTeacherOracle"]
