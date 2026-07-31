"""Launch the ALOD metrics dashboard with a Windows SSL-store fallback.

On some Windows Python installs, importing Streamlit/Tornado can fail before
the dashboard script runs because ``ssl.create_default_context`` cannot parse a
bad certificate entry from the Windows certificate store. This launcher installs
a narrow fallback before importing Streamlit, then delegates to
``streamlit run tools/view_metrics.py``.
"""

from __future__ import annotations

import os
import ssl
import sys
from pathlib import Path
from typing import Any


def _fallback_ssl_context(purpose: Any) -> ssl.SSLContext:
    protocol = (
        ssl.PROTOCOL_TLS_SERVER
        if purpose == ssl.Purpose.CLIENT_AUTH
        else ssl.PROTOCOL_TLS_CLIENT
    )
    context = ssl.SSLContext(protocol)
    if protocol == ssl.PROTOCOL_TLS_CLIENT:
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True

    cafile = os.environ.get('SSL_CERT_FILE')
    capath = os.environ.get('SSL_CERT_DIR')
    if cafile or capath:
        context.load_verify_locations(cafile=cafile, capath=capath)
        return context

    try:
        import certifi  # type: ignore
    except ImportError:
        certifi = None
    if certifi is not None:
        context.load_verify_locations(cafile=certifi.where())
        return context

    context.set_default_verify_paths()
    return context


def _install_windows_ssl_store_fallback() -> None:
    original_create_default_context = ssl.create_default_context

    def create_default_context(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        try:
            return original_create_default_context(*args, **kwargs)
        except ssl.SSLError as exc:
            message = str(exc)
            if 'ASN1' not in message and 'NOT_ENOUGH_DATA' not in message:
                raise
            purpose = (
                args[0]
                if args else kwargs.get('purpose', ssl.Purpose.SERVER_AUTH)
            )
            return _fallback_ssl_context(purpose)

    ssl.create_default_context = create_default_context


def main() -> None:
    _install_windows_ssl_store_fallback()
    dashboard_script = Path(__file__).with_name('view_metrics.py')
    sys.argv = [
        'streamlit',
        'run',
        str(dashboard_script),
        '--',
        *sys.argv[1:],
    ]
    from streamlit.web import cli as streamlit_cli

    streamlit_cli.main()


if __name__ == '__main__':
    main()

