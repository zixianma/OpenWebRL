"""Keep software-rendered local browsers out of the NVIDIA EGL driver."""
import os
from pathlib import Path


MESA_EGL_VENDOR = Path('/usr/share/glvnd/egl_vendor.d/50_mesa.json')


def browser_process_environment(browser_args, *, environ=None, mesa_vendor=MESA_EGL_VENDOR):
    """Return a child-only environment, respecting explicit EGL vendor settings.

    Chromium's --disable-gpu still allows EGL device enumeration. On H200 nodes,
    concurrent browser GPU processes can block in NVIDIA modeset calls before
    their first frame. Restrict EGL discovery to Mesa for software browsers when
    its vendor manifest exists. Do not change the actor's process environment or
    CUDA visibility; hardware-rendered browsers retain their original setup.
    """
    child = dict(os.environ if environ is None else environ)
    if ('--disable-gpu' in browser_args
            and '__EGL_VENDOR_LIBRARY_FILENAMES' not in child
            and Path(mesa_vendor).is_file()):
        child['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(mesa_vendor)
    return child
