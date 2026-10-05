"""Conservative CUDA cubin compatibility, without importing torch or starting GPU work."""
import re


def supports_device(major, minor, compiled_archs):
    """Accept a native cubin for the same major and an equal or lower minor capability.

    NVIDIA permits sm_86 cubins on sm_89, so an exact architecture-string match is too strict.
    PTX-only targets and architecture-specific suffixes are not proof of native cubin support.
    Missing or unreadable device/build information cannot establish compatibility.
    """
    if type(major) is not int or type(minor) is not int or major < 1 or not 0 <= minor <= 9:
        return False
    if not isinstance(compiled_archs, (list, tuple)):
        return False
    for arch in compiled_archs:
        match = re.fullmatch(r"sm_([1-9][0-9]{1,2})", arch) if isinstance(arch, str) else None
        if match:
            capability = int(match.group(1))
            if capability // 10 == major and capability % 10 <= minor:
                return True
    return False
