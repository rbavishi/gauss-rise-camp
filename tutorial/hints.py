import attr
import os
from IPython.display import display, Image, Markdown
from ipywidgets import widgets


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


@attr.s(cmp=False, repr=False)
class ShowHideWidget:
    title: str = attr.ib()
    base_widget = attr.ib()

    _output_widget = attr.ib(init=False)
    _toggle_button = attr.ib(init=False)
    _value = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._output_widget = widgets.Output()
        with self._output_widget:
            display(self.base_widget)

        self._toggle_button = widgets.Button(layout=widgets.Layout(width='auto'),
                                             description=self.title,
                                             disabled=False,
                                             button_style='',
                                             tooltip='Show',
                                             icon='caret-down')

        self._toggle_button.observe(self._onclick, names=['value'])
        self._toggle_button.on_click(self._onclick)
        _value = False

    def display(self, initial_show: bool = False):
        if initial_show:
            self.show()
        else:
            self.hide()

        display(self._toggle_button)
        display(self._output_widget)

    def show(self):
        self._value = True
        self._output_widget.layout.display = ''
        self._toggle_button.icon = 'caret-up'
        self._toggle_button.tooltip = "Hide"

    def hide(self):
        self._value = False
        self._output_widget.layout.display = 'none'
        self._toggle_button.icon = 'caret-down'
        self._toggle_button.tooltip = "Show"

    def _onclick(self, _):
        if self._value is True:
            self.hide()

        else:
            self.show()


_demo_1_hints = """
<ul>
    <li>You only need to provide one cell in the output - one of the sums.</li>
</ul>
<img src="gifs/Solution-Demo-1.gif" width="60%" height="60%">
"""


_demo_2_hints = """
<ul>
    <li>For obtaining column subtraction, one cell value is enough. However, this won't delete the Low and High columns.</li>
    <li>You also need to mark Low and High as deleted.</li>
</ul>
<img src="gifs/Solution-Demo-2.gif" width="60%" height="60%">
"""


_ex_1_hints = """
<ul>
    <li>Getting lots of nans? Try providing more of the output. In particular, provide the new column values (metrics) as well as a column containing the years.</li>
</ul>
<img src="gifs/Solution-Ex-1.gif" width="60%" height="60%">
"""


_ex_2_hints = """
<ul>
    <li>Use the STR-SPLIT operation to separate column names into days and weeks.</li> 
    <li>Signal that you want to retain the Plants column by copying Plants onto a column in the output.</li>
</ul>
<img src="gifs/Solution-Ex-2.gif" width="50%" height="50%">
"""


HINTS_DICT = {
    'demo-1': _demo_1_hints,
    'demo-2': _demo_2_hints,
    'ex-1': _ex_1_hints,
    'ex-2': _ex_2_hints,
}


def setup(problem_id: str):
    output = widgets.Output()
    with output:
        display(Markdown(HINTS_DICT[problem_id]))

    w = ShowHideWidget("Show Hints", output)
    w.display()

