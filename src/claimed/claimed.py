import importlib
import inspect
import os
import subprocess
import sys


def _parse_kwargs(rest, sig):
    """Parse --key value pairs from a list of CLI tokens, coerce types via signature."""
    kwargs = {}
    i = 0
    while i < len(rest):
        token = rest[i]
        if token.startswith('--'):
            key = token[2:].replace('-', '_')
            if i + 1 < len(rest) and not rest[i + 1].startswith('--'):
                kwargs[key] = rest[i + 1]
                i += 2
            else:
                # bare flag → True
                kwargs[key] = True
                i += 1
        else:
            i += 1

    # Coerce string values to the expected type using annotation or default
    for name, param in sig.parameters.items():
        if name not in kwargs:
            continue
        val = kwargs[name]
        if not isinstance(val, str):
            continue
        # Try annotation first
        ann = param.annotation
        if ann is not inspect.Parameter.empty:
            target = ann
            if hasattr(ann, '__origin__'):
                # e.g. Optional[X] – skip complex generics
                continue
            try:
                kwargs[name] = target(val)
                continue
            except Exception:
                pass
        # Fall back to the type of the default value
        default = param.default
        if default is not inspect.Parameter.empty and default is not None:
            try:
                kwargs[name] = type(default)(val)
            except Exception:
                pass

    return kwargs


def _run_module(args):
    if not args:
        print("Usage: claimed run <module.path> [--param-name value ...] [--help]")
        sys.exit(1)

    module_path = args[0]
    rest = args[1:]

    # Import the target module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"Error: cannot import module '{module_path}': {e}")
        sys.exit(1)

    if not hasattr(module, 'run'):
        print(f"Error: module '{module_path}' has no 'run' function.")
        sys.exit(1)

    fn = module.run
    sig = inspect.signature(fn)

    # --help: print signature and docstring
    if '--help' in rest:
        print(f"Module : {module_path}")
        print(f"Function: {module_path}.run{sig}")
        doc = inspect.getdoc(fn)
        if doc:
            print(f"\n{doc}\n")
        print("Parameters:")
        for pname, param in sig.parameters.items():
            flag = '--' + pname.replace('_', '-')
            ann = param.annotation
            type_hint = (
                ann.__name__ if (ann is not inspect.Parameter.empty and hasattr(ann, '__name__'))
                else str(ann) if ann is not inspect.Parameter.empty
                else 'any'
            )
            default = (
                f'  (default: {param.default!r})'
                if param.default is not inspect.Parameter.empty
                else '  (required)'
            )
            print(f"  {flag}  <{type_hint}>{default}")
        sys.exit(0)

    kwargs = _parse_kwargs(rest, sig)
    fn(**kwargs)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        _run_module(sys.argv[2:])
        return

    dir_path = os.path.dirname(os.path.realpath(__file__))
    return subprocess.call(
        f'{dir_path}/scripts/claimed ' + ' '.join(sys.argv[1:]), shell=True
    )


if __name__ == '__main__':
    main()

