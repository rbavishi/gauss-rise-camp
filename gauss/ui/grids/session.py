import json
import sys
import traceback
import uuid
from typing import List, Any, Dict, Union, Optional, Callable

import attr
import numpy as np
import pandas as pd
from IPython.display import display, clear_output, Javascript, HTML
from ipywidgets import widgets

from pygments import highlight
from pygments import lexers
from pygments.formatters.html import HtmlFormatter
from pygments.styles import get_style_by_name

from gauss.synthesis.problem import SynthesisProblem
from gauss.ui.grids import utils
from gauss.ui.grids.common import GAUSS_MAGIC_PREFIX
from gauss.ui.grids.widgets import InputGridWidget, OutputGridWidget, ScratchGridWidget, SolutionGridWidget
from gauss.ui.session import UISession


def _highlight_code(code: str):
    lexer = lexers.get_lexer_by_name('python')
    style = get_style_by_name('xcode')
    formatter = HtmlFormatter(full=True, style=style)
    return HTML(highlight(code, lexer, formatter))


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


@attr.s(cmp=False, repr=False)
class SynthesisOutputWidget:
    session: UISession = attr.ib()
    allow_composition: bool = attr.ib(default=True)

    _on_composition: Callable = attr.ib(default=None)
    _engine_iter = attr.ib(init=False, default=None)
    _code_stack: List[str] = attr.ib(init=False, default=None)

    _overall_output = attr.ib(init=False, default=None)
    _buttons_output = attr.ib(init=False, default=None)
    _solution_output = attr.ib(init=False, default=None)
    _msg_output = attr.ib(init=False, default=None)
    _code_output = attr.ib(init=False, default=None)

    _prev_button = attr.ib(init=False, default=None)
    _next_button = attr.ib(init=False, default=None)
    _use_as_input_button = attr.ib(init=False, default=None)
    _explanation_widget = attr.ib(init=False, default=None)

    _index: int = attr.ib(init=False, default=None)
    _solutions = attr.ib(init=False, default=None)
    _solution_grid: SolutionGridWidget = attr.ib(init=False, default=None)
    _solution_grid_widget = attr.ib(init=False, default=None)
    _stopped: bool = attr.ib(init=False, default=None)

    def __attrs_post_init__(self):
        self._overall_output = widgets.Output()
        self._buttons_output = widgets.Output()
        self._solution_output = widgets.Output()
        self._msg_output = widgets.Output()
        self._code_output = widgets.Output()
        grid_widget = widgets.Output()

        self._next_button = widgets.Button(description='Next')
        self._prev_button = widgets.Button(description='Prev')
        self._use_as_input_button = widgets.Button(description='Use As Input')

        self._next_button.on_click(self._onclick_next)
        self._prev_button.on_click(self._onclick_prev)
        self._use_as_input_button.on_click(self._onclick_use_as_input)

        self._index = 0
        self._solutions = []
        self._stopped = False
        self._solution_grid = SolutionGridWidget(self.session, grid_id="A")
        self._solution_grid_widget = self._solution_grid.get_widget()
        self._explanation_widget = widgets.Text(description=r'\(f\)',
                                                disabled=True,
                                                style={'description_width': '9px'},
                                                )

        with self._buttons_output:
            display(widgets.HBox([self._prev_button, self._next_button]))

        with self._solution_output:
            display(grid_widget)
            display(self._code_output)
            if self.allow_composition:
                display(self._use_as_input_button)

        with grid_widget:
            display(self._explanation_widget)
            display(self._solution_grid_widget)

        with self._overall_output:
            display(self._buttons_output)
            display(self._solution_output)
            display(self._msg_output)

    def display(self):
        display(self._overall_output)

    def show(self):
        self._overall_output.layout.display = ''

    def hide(self):
        self._overall_output.layout.display = 'none'

    def _show_solution(self):
        self._solution_output.layout.display = ''
        self._msg_output.layout.display = 'none'

    def _show_msg(self):
        self._solution_output.layout.display = 'none'
        self._msg_output.layout.display = ''

    def _show_buttons(self):
        self._buttons_output.layout.display = ''

    def _hide_buttons(self):
        self._buttons_output.layout.display = 'none'

    def run(self, engine_iter, code_stack, on_composition: Callable = None):
        self._engine_iter = engine_iter
        self._code_stack = code_stack
        self._on_composition = on_composition

        self._index = 0
        self._solutions = []
        self._code_output.clear_output()
        self._msg_output.clear_output()
        self._stopped = False
        self._hide_buttons()
        self._display_msg('Searching...')
        self._display_current_solution()
        if len(self._solutions) > 0:
            self._show_buttons()

    def explain(self, row, col):
        with self._overall_output:
            text = self._solution_grid.get_explanation(row, col)
            if text is not None:
                self._explanation_widget.value = text
            else:
                self._explanation_widget.value = ''

    def _display_current_solution(self):
        if self._stopped and self._index >= (len(self._solutions) - 1):
            self._display_msg(self._solutions[-1])
            self._index = len(self._solutions) - 1
            return

        while self._index >= len(self._solutions):
            try:
                self._solutions.append(next(self._engine_iter))

            except TimeoutError:
                if self._index == 0:
                    msg = "No solutions found within 10 seconds. Try providing more information about the output."
                    self._solutions.append(msg)
                    self._stopped = True
                    break

                else:
                    msg = "No more solutions found within 10 seconds."
                    self._solutions.append(msg)
                    self._stopped = True
                    break

            except StopIteration:
                if self._index == 0:
                    msg = "No solutions found. Verify if your output is correct."
                    self._solutions.append(msg)
                    self._stopped = True
                    break

                else:
                    msg = "No more solutions found."
                    self._solutions.append(msg)
                    self._stopped = True
                    break

        if self._stopped and self._index >= (len(self._solutions) - 1):
            self._display_msg(self._solutions[-1])
            self._index = len(self._solutions) - 1
            return

        solution = self._solutions[self._index]
        code = solution.code
        produced_output = solution.output
        prev_code = '\n'.join(self._code_stack)
        code = f"{prev_code}\n{code}"
        self._explanation_widget.value = ''

        with self._code_output:
            clear_output()
            display(_highlight_code(code))

        self._show_solution()
        self._solution_grid.update(produced_output, explanations=solution.explanations)

    def _display_msg(self, msg):
        with self._msg_output:
            clear_output()
            print(msg)

        self._show_msg()

    def _onclick_next(self, _):
        self._index += 1
        self._display_current_solution()

    def _onclick_prev(self, _):
        if self._index > 0:
            self._index -= 1
            self._display_current_solution()

    def _onclick_use_as_input(self, _):
        solution = self._solutions[self._index]
        code = solution.code
        produced_output = solution.output
        self._on_composition(code=code, output=produced_output)


@attr.s(cmp=False, repr=False)
class GridUISession(UISession):
    _input_grids: List[InputGridWidget] = attr.ib(init=False, factory=list)
    _output_grid: OutputGridWidget = attr.ib(init=False)
    _scratch_grid: ScratchGridWidget = attr.ib(init=False)
    _synthesis_output_widget: SynthesisOutputWidget = attr.ib(init=False)

    _wrappers_input_grids: List[ShowHideWidget] = attr.ib(init=False, factory=list)
    _wrapper_output_grid: ShowHideWidget = attr.ib(init=False)
    _wrapper_scratch_grid: ShowHideWidget = attr.ib(init=False)

    _computation_history = attr.ib(init=False)

    _synthesize_button = attr.ib(init=False)
    _synthesize_output = attr.ib(init=False)
    _synthesize_solution_output = attr.ib(init=False)

    _output_reset_button = attr.ib(init=False)

    _obj_store: Dict[str, Any] = attr.ib(init=False, factory=dict)
    _value_traces: Dict[str, Dict] = attr.ib(init=False, factory=dict)
    _value_trackers: Dict[str, str] = attr.ib(init=False, factory=dict)
    _interactions: List[Dict] = attr.ib(init=False, factory=list)
    _constants: List[Any] = attr.ib(init=False, factory=list)

    _cur_output_name: str = attr.ib(init=False, default='')
    _cur_code_stack: List[str] = attr.ib(init=False, factory=list)

    _use_scratch: bool = attr.ib(default=False)
    _allow_composition: bool = attr.ib(default=True)

    _cur_solution_grid: SolutionGridWidget = attr.ib(init=False, default=None)
    _explanation_widget = attr.ib(init=False, default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.reset()

    def reset(self):
        self._input_grids = []
        self._wrappers_input_grids = []
        self._obj_store = {}
        self._value_traces = {}
        self._value_trackers = {}
        self._interactions = []
        self._constants = []

        for i, obj in enumerate(self.inputs):
            grid_id = f"I{i}"
            value_trackers, value_traces = self._get_value_trackers_and_traces(obj, grid_id)
            self._value_traces.update(value_traces)
            self._value_trackers.update(value_trackers)
            self._input_grids.append(InputGridWidget(session=self, grid_id=grid_id,
                                                     df=obj,
                                                     value_trackers=value_trackers))
            self._wrappers_input_grids.append(ShowHideWidget(f"Input {i + 1}", self._input_grids[-1].get_widget()))

        self._output_grid = OutputGridWidget(grid_id="O", session=self)
        self._scratch_grid = ScratchGridWidget(grid_id="S", session=self)
        self._synthesis_output_widget = SynthesisOutputWidget(session=self,
                                                              allow_composition=self._allow_composition)

        self._wrapper_output_grid = ShowHideWidget('Partial Output', self._output_grid.get_widget())
        self._wrapper_scratch_grid = ShowHideWidget('Scratch', self._scratch_grid.get_widget())

        self._synthesize_button = widgets.Button(description='Synthesize')
        self._synthesize_button.button_style = 'success'
        self._synthesize_output = widgets.Output()
        self._synthesize_solution_output = widgets.Output()

        self._synthesize_button.on_click(self._onclick_synthesize)

        self._output_reset_button = widgets.Button(description="Reset")

        self._output_reset_button.on_click(self._reset_output)
        self._output_reset_button.button_style = 'info'

        for i, obj in enumerate(self.inputs):
            self._obj_store[f"I{i}"] = obj

        self._explanation_widget = widgets.Text(description=r'\(f\)',
                                                disabled=True,
                                                style={'description_width': '9px'},
                                                )

    def display(self):
        for idx, grid in enumerate(self._wrappers_input_grids, 1):
            grid.display(initial_show=True)

        if self._use_scratch:
            self._wrapper_scratch_grid.display(initial_show=False)

        self._wrapper_output_grid.display(initial_show=True)
        self._synthesis_output_widget.hide()

        display(widgets.HBox([self._synthesize_button, self._output_reset_button]))
        self._synthesis_output_widget.display()

    def _reset_output(self, *args, **kwargs):
        self._notify("All interactions have been reset.", mode='info')
        self._interactions.clear()
        self._synthesize_solution_output.clear_output()
        self._synthesize_output.clear_output()

        self._output_grid = OutputGridWidget(grid_id="O", session=self)
        self._wrapper_output_grid = ShowHideWidget('Partial Output', self._output_grid.get_widget())

        with self.output_widget:
            clear_output()

        with self.output_widget:
            self.display()

    def _get_interactions(self):
        return self._interactions

    def process_computation(self, args: Union[str, dict]):
        try:
            if isinstance(args, str):
                args = json.loads(args)

            other_args = args.copy()
            other_args.pop('operation')
            other_args.pop('cells')

            #  Handle builtins
            if args['operation'] == "DELETE":
                trace = {"from": args['cells'], "labels": ["DELETE"], "to": "", "value": None}
                self._interactions.append(trace)
                return

            result = self.domain_ui.perform_operation(args['operation'],
                                                      [(self._value_traces[tracker],
                                                        self._value_traces[tracker]['value'])
                                                       for tracker in args['cells']],
                                                      self._obj_store,
                                                      {k: self._value_traces[v]
                                                       for k, v in self._value_trackers.items()},
                                                      other_args)

            values = result['traces']
            if 'constants' in result:
                self._constants.extend(result['constants'])

            value_strs = []
            for row in values:
                value_strs.append([])
                for elem in row:
                    value = elem['value']
                    tracker = self._gen_uid()
                    trace = elem.copy()
                    trace["tracker"] = tracker
                    trace["from"] = elem["from"][:]
                    self._value_traces[tracker] = trace
                    value_strs[-1].append(f"{GAUSS_MAGIC_PREFIX}{tracker}{GAUSS_MAGIC_PREFIX}"
                                          f"{value}")

            value_str = "\n".join("\t".join(row_value_strs) for row_value_strs in value_strs)

            print(utils.prepare_json_string(json.dumps({
                "success": "true",
                "value": value_str,
            })))

        except Exception as e:
            print(utils.prepare_json_string(json.dumps({
                "success": "false",
                "msg": traceback.format_exc(),
            })))

    def record_interaction(self, tracker: str, out_row: str, out_column: str):
        trace = self._value_traces[tracker]
        interaction = trace.copy()
        interaction["to"] = f"{out_row}:{out_column}:O"

        self._interactions.append(interaction)

    def _get_value_trackers_and_traces(self, df: pd.DataFrame, grid_id: str):
        if df.columns.nlevels != 1 or df.index.nlevels != 1:
            raise AssertionError("Multi-index not supported.")

        value_trackers: Dict[str, str] = {}
        value_traces: Dict[str, Dict] = {}

        for idx, col in enumerate(df.columns):
            tracker = self._gen_uid()
            key = f"-1:{idx}:{grid_id}"
            value_trackers[key] = tracker
            value_traces[tracker] = {
                "from": [key],
                "labels": ["EQUAL"],
                "value": col,
                "tracker": tracker,
            }

        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                val = df.iloc[r, c]
                tracker = self._gen_uid()
                key = f"{r}:{c}:{grid_id}"

                value_trackers[key] = tracker
                value_traces[tracker] = {
                    "from": [key],
                    "labels": ["EQUAL"],
                    "value": val,
                    "tracker": tracker,
                }

        return value_trackers, value_traces

    def _gen_uid(self):
        return uuid.uuid4().hex

    def _find_used_inputs(self, interaction):
        used = []
        for f in interaction['from']:
            if isinstance(f, str):
                used.append(f.split(':')[-1])
            else:
                used.extend(self._find_used_inputs(f))

        return used

    def _onclick_synthesize(self, *args, **kwargs):
        if len(self._interactions) == 0:
            self._notify("No interactions have been performed.", mode='error')
            return

        used_inputs = set()
        for interaction in self._interactions:
            used_inputs.update(self._find_used_inputs(interaction))

        input_names = []
        main_module = sys.modules["__main__"]
        for idx, inp in enumerate(self.inputs, 1):
            if f"I{idx - 1}" not in used_inputs:
                continue

            for k, v in main_module.__dict__.items():
                if inp is v and (not k.startswith("_")):
                    input_names.append(k)
                    break
            else:
                input_names.append(f"inp{idx}")

        idx = 0
        output_name = ''
        while idx >= 0:
            output_name = f"out_{idx}"
            if output_name not in main_module.__dict__:
                break

            idx += 1

        output, graph, g_inputs = self.domain_ui.process_ui_interaction({f"I{i}": inp
                                                                         for i, inp in enumerate(self.inputs)},
                                                                        self._get_interactions())

        self._cur_output_name = output_name
        problem = SynthesisProblem([inp for i, inp in enumerate(self.inputs) if f"I{i}" in used_inputs],
                                   output, graph,
                                   [g_inputs[f"I{i}"] for i in range(len(self.inputs)) if f"I{i}" in used_inputs],
                                   constants=self._constants,
                                   check_strict=False,
                                   input_names=input_names,
                                   output_name=output_name,
                                   timeout=10)

        engine_iter = self.engine.solve(problem)

        def on_composition(code: str, output: Any):
            sys.modules["__main__"].__dict__[self._cur_output_name] = output
            self.inputs.append(output)
            self._cur_code_stack.append(code)

            with self.output_widget:
                clear_output()

            self.reset()
            with self.output_widget:
                self.display()

        self._wrapper_output_grid.hide()
        self._synthesis_output_widget.show()
        self._synthesis_output_widget.run(engine_iter, self._cur_code_stack, on_composition=on_composition)

    def _notify(self, text: str, mode='error'):
        if mode == 'error':
            code = Javascript(f"""$.notify("Error : {text}", "error", 
                                     $.notify.defaults({{className: "error", "position": "bottom center"}}));""")

        elif mode == 'info':
            code = Javascript(f"""$.notify("{text}", "info", 
                                     $.notify.defaults({{className: "info", "position": "bottom center"}}));""")

        display(code)

    def _clear_output_cells(self, cells):
        cells = set(cells)
        self._interactions = [i for i in self._interactions if i['to'] not in cells]

    def _onclick_scratch(self, change):
        if change['new']:
            self._scratch_output.layout.display = ''
            self._scratch_toggle.icon = 'caret-up'
            self._scratch_toggle.description = "Hide Scratch"

        else:
            self._scratch_output.layout.display = 'none'
            self._scratch_toggle.icon = 'caret-down'
            self._scratch_toggle.description = "Show Scratch"

    def explain(self, row, col):
        self._synthesis_output_widget.explain(row, col)
