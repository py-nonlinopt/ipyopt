import subprocess
import warnings


def pkg_config(*packages, **kwargs):
    """Calls pkg-config returning a dict containing all arguments
    for Extension() needed to compile the extension
    """
    flag_map = {b'-I': 'include_dirs',
                b'-L': 'library_dirs',
                b'-l': 'libraries',
                b'-D': 'define_macros'}
    res = subprocess.run(
        ("pkg-config", "--libs", "--cflags")
        + packages, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    if res.stderr:
        raise RuntimeError(res.stderr.decode())
    for token in res.stdout.split():
        kwargs.setdefault(flag_map.get(token[:2]), []).append(
            token[2:].decode())
    define_macros = kwargs.get('define_macros')
    if define_macros:
        kwargs['define_macros'] = [tuple(d.split()) for d in define_macros]
    undefined_flags = kwargs.pop(None, None)
    if undefined_flags:
        warnings.warn(
            "Ignoring flags {} from pkg-config".format(", ".join(undefined_flags)))
    return kwargs
