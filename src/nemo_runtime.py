"""Small NeMo runtime configuration shared by inference and evaluation."""

from __future__ import annotations

import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO


def silence_nemo_configuration_logs() -> None:
    """Keep errors visible while hiding NeMo setup/configuration diagnostics."""
    from nemo.utils import logging as nemo_logging

    nemo_logging.setLevel(logging.ERROR)
    for name in (
        "nemo",
        "nemo_logger",
        "nv_one_logger",
        "pytorch_lightning",
        "lightning",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


@contextmanager
def quiet_nemo_transcribe():
    """Suppress NeMo's non-actionable stderr diagnostics for one inference call."""
    silence_nemo_configuration_logs()
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        yield
