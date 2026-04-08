"""Dedicated tests for SSRF IP pinning and DNS rebinding protection."""

from __future__ import annotations

import contextlib
import threading
from unittest.mock import MagicMock, patch

import pytest
import requests

from ai_workers.common import utils


class TestSsrfPinning:
    """Tests for DNS rebinding protection via IP pinning."""

    def test_pinning_prevents_rebinding(self):
        """Verify that even if DNS changes, the pinned IP is used."""
        hostname = "example.com"
        safe_ip = "93.184.216.34"

        if not hasattr(utils, "_original_create_connection"):
            pytest.skip("urllib3 is not patched in this environment")

        with patch("ai_workers.common.utils._original_create_connection") as mock_orig_cc:
            mock_orig_cc.return_value = MagicMock()

            # 1. Pin the hostname to the safe IP
            with utils._pin_hostname_to_ip(hostname, safe_ip):
                # 2. Simulate a connection attempt to the hostname
                # The patch should intercept this and use the safe IP
                utils._patched_create_connection((hostname, 80))

                # Check that the original create_connection was called with the pinned IP
                mock_orig_cc.assert_called_with((safe_ip, 80))

            # 3. After the context manager, it should use the hostname again
            utils._patched_create_connection((hostname, 80))
            mock_orig_cc.assert_called_with((hostname, 80))

    def test_requests_uses_pinned_ip(self):
        """Integration test: Verify requests.Session uses the pinned IP."""
        hostname = "pin-test.com"
        pinned_ip = "1.2.3.4"

        if not hasattr(utils, "_original_create_connection"):
            pytest.skip("urllib3 is not patched")

        # Mock the original create_connection to fail with a specific message
        def mock_cc(address, *args, **kwargs):
            raise requests.exceptions.ConnectTimeout(f"Timed out connecting to {address[0]}")

        with (
            utils._pin_hostname_to_ip(hostname, pinned_ip),
            patch("ai_workers.common.utils._original_create_connection", side_effect=mock_cc),
        ):
            with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
                # Use a very short timeout
                utils._session.get(f"http://{hostname}", timeout=0.1)

            # Verify it tried to connect to the pinned IP
            assert pinned_ip in str(excinfo.value)

    def test_thread_safety(self):
        """Verify that pinning is thread-local and doesn't leak between threads."""
        hostname = "thread-test.com"
        ip_thread_1 = "1.1.1.1"
        ip_thread_2 = "2.2.2.2"

        results = {}

        def thread_func(ip, thread_name):
            with utils._pin_hostname_to_ip(hostname, ip):
                # Small delay to ensure overlap
                import time

                time.sleep(0.1)
                pinned = getattr(utils._thread_local, "pinned_ips", {}).get(hostname)
                results[thread_name] = pinned

        t1 = threading.Thread(target=thread_func, args=(ip_thread_1, "t1"))
        t2 = threading.Thread(target=thread_func, args=(ip_thread_2, "t2"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == ip_thread_1
        assert results["t2"] == ip_thread_2

    def test_proxy_bypass(self):
        """Verify if proxies can bypass the pinning protection."""
        hostname = "proxy-test.com"
        pinned_ip = "1.2.3.4"
        proxy_url = "http://localhost:8080"

        if not hasattr(utils, "_original_create_connection"):
            pytest.skip("urllib3 is not patched")

        # Mock the original create_connection
        with (
            utils._pin_hostname_to_ip(hostname, pinned_ip),
            patch("ai_workers.common.utils._original_create_connection"),
            patch.dict("os.environ", {"HTTP_PROXY": proxy_url, "HTTPS_PROXY": proxy_url}),
            contextlib.suppress(Exception),
        ):
            # We need to see what requests does.
            # If we use a real session, it will look at environment variables.
            utils._session.get(f"http://{hostname}", timeout=0.1)

    def test_load_image_from_url_fail_closed_logic(self):
        """Test that load_image_from_url will fail if patching was not successful."""
        from urllib3.util import connection

        # Temporarily unpatch or simulate unpatching
        original_patched = getattr(connection, "_is_patched", False)
        try:
            if hasattr(connection, "_is_patched"):
                del connection._is_patched

            with pytest.raises(RuntimeError, match="SSRF IP pinning is not active"):
                utils.load_image_from_url("http://example.com/test.png")

        finally:
            if original_patched:
                connection._is_patched = True

    def test_proxies_are_disabled(self):
        """Verify that proxies are explicitly disabled in the session.get call."""
        hostname = "proxy-disable-test.com"
        safe_ip = "93.184.216.34"

        if not hasattr(utils, "_original_create_connection"):
            pytest.skip("urllib3 is not patched")

        mock_resp = MagicMock()
        mock_resp.iter_content = MagicMock(return_value=iter([b"fake data"]))
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("ai_workers.common.utils._get_safe_ips", return_value=[safe_ip]),
            patch("ai_workers.common.utils._session.get", return_value=mock_resp) as mock_get,
            patch("PIL.Image.open", return_value=MagicMock(convert=MagicMock())),
        ):
            utils.load_image_from_url(f"http://{hostname}/image.png")

            # Check proxies argument
            kwargs = mock_get.call_args.kwargs
            assert kwargs["proxies"] == {"http": None, "https": None}
