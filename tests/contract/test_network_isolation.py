"""Guard the default test suite against accidental network access."""

import socket

import pytest
from pytest_socket import SocketBlockedError


@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket.:UserWarning")
def test_default_test_suite_blocks_network_sockets() -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket()
