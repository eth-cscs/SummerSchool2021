import tensorboard
from IPython import get_ipython


def _display_ipython(port, height, display_handle):
    import os
    import IPython.display
    iframe = IPython.display.IFrame(
        src="https://%s.jupyter.cscs.ch/hub/user-redirect/proxy/%d/#scalars" % (os.environ['USER'], port),
        # src="https://%s.jupyter.cscs.ch/user/sarafael/tensorboard" % (os.environ['USER']),
        height=height,
        width="100%",
    )
    if display_handle:
        display_handle.update(iframe)
    else:
        IPython.display.display(iframe)


ipy = get_ipython()
tensorboard.notebook._load_ipython_extension(ipy)

# monkey patching the `_display_ipython` function to change the display URL
tensorboard.notebook._display_ipython = _display_ipython
