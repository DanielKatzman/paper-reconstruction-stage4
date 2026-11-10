"""Process-local crash logging before native libraries are imported.

No watchdog thread or timeout is installed. Prefer Windows error reporting
without modal UI; the Win32 fallback retains native stderr tracebacks.
"""
import faulthandler
import sys


def configure():
    status = {'native_tracebacks': False, 'timer_watchdog_installed': False,
              'windows_error_ui_disabled': False}
    try:
        faulthandler.enable(all_threads=True)
        status['native_tracebacks'] = True
    except (OSError, ValueError, RuntimeError):
        pass  # Some embedded invocations have no real stderr descriptor.
    if sys.platform == 'win32':
        try:
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            get_flags = kernel.WerGetFlags
            get_flags.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            get_flags.restype = ctypes.c_long
            set_flags = kernel.WerSetFlags
            set_flags.argtypes = [wintypes.DWORD]
            set_flags.restype = ctypes.c_long
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            flags = wintypes.DWORD()
            result = get_flags(kernel.GetCurrentProcess(), ctypes.byref(flags))
            # WER may have no entry until this process first sets its flags.
            if result == 0 or (result & 0xffffffff) == 0x80070490:
                result = set_flags(flags.value | 0x20)
                status['windows_error_ui_disabled'] = result == 0
            if not status['windows_error_ui_disabled']:
                status['windows_wer_configuration_error'] = f'0x{result & 0xffffffff:08X}'
                # Some restricted processes cannot change WER flags. Keep the
                # native stderr handler and use the process-local Win32 fallback.
                kernel.GetErrorMode.restype = wintypes.UINT
                kernel.SetErrorMode.argtypes = [wintypes.UINT]
                kernel.SetErrorMode.restype = wintypes.UINT
                kernel.SetErrorMode(kernel.GetErrorMode() | 0x0001 | 0x0002)
                status['windows_error_ui_disabled'] = bool(kernel.GetErrorMode() & 0x0002)
                status['windows_error_mode_fallback'] = True
        except (AttributeError, OSError):
            pass
    return status
