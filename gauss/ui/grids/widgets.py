from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional

import attr
import pandas as pd
from IPython.core.display import display, Javascript
from ipyaggrid import Grid

from gauss.ui.grids.common import NOTIFY_JS
from gauss.ui.grids.helpers import gen_js_helpers
from gauss.ui.grids.interaction import genScratchClipboardCallbackForCell, genOutputClipboardCallbackCell, \
    genOutputClipboardCallbackData, genScratchClipboardCallbackFromCell, genScratchClipboardCallbackFromData, \
    genContextMenuNonOutput, genContextMenuOutput
from gauss.ui.session import UISession
from gauss.graphs import Graph

COLUMN_ROW_RENDERER = """
function(params) {
    let row = helpers.convRow(params.node.id);
    let col = helpers.convCol(params.colDef.field);
    let textDecoration = '';
    let isDeleted = helpers.isMarkedForDeletion(row, col);
    let isSelected = helpers.isSelected(row, col);
   
    if (params.node.id === "0") {
        if (isSelected) {
            if (isDeleted) {
                return {backgroundColor: 'darkred', color: 'white', textDecoration: 'line-through'};
            } else {
                return {backgroundColor: 'darkblue', color: 'white', textDecoration: ''};
            }
        } else {
            if (isDeleted) {
                return {backgroundColor: '#ff9999', color: 'black', textDecoration: 'line-through'};
            } else {
                return {backgroundColor: 'lightblue', color: 'black', textDecoration: ''};
            }
        }
    } else {
        if (isSelected) {
            if (isDeleted) {
                return {backgroundColor: '#ff8080', color: 'white', textDecoration: 'line-through'};
            } 
            else {
                return {backgroundColor: '#999999', color: 'black', textDecoration: ''};
            }
        } else {
            if (isDeleted) {
                return {
                    backgroundColor: '#ffcccc',
                    color: 'black', textDecoration: 'line-through'
                };
            } else {
                return {
                    backgroundColor: (parseInt(row) % 2 == 0) ? '#F9F9F9' : '#FFFFFF',
                    color: 'black', textDecoration: ''
                };
            }
        }
    } 
}""".strip()

COLUMN_ROW_RENDERER_OUTPUT = """
function(params) {
    let row = helpers.convRow(params.node.id);
    let col = helpers.convCol(params.colDef.field);
    let textDecoration = '';
    let isDeleted = helpers.isMarkedForDeletion(row, col);
    let isSelected = helpers.isSelected(row, col);

    if (params.node.id === "0") {
        if (isSelected) {
            if (isDeleted) {
                return {backgroundColor: 'darkred', color: 'white', textDecoration: 'line-through'};
            } else {
                return {backgroundColor: 'darkorange', color: 'white', textDecoration: ''};
            }
        } else {
            if (isDeleted) {
                return {backgroundColor: '#ff9999', color: 'black', textDecoration: 'line-through'};
            } else {
                return {backgroundColor: '#fce5cd', color: 'black', textDecoration: ''};
            }
        }
    } else {
        if (isSelected) {
            if (isDeleted) {
                return {backgroundColor: '#ff8080', color: 'white', textDecoration: 'line-through'};
            } 
            else {
                return {backgroundColor: '#999999', color: 'black', textDecoration: ''};
            }
        } else {
            if (isDeleted) {
                return {
                    backgroundColor: '#ffcccc',
                    color: 'black', textDecoration: 'line-through'
                };
            } else {
                return {
                    backgroundColor: (parseInt(row) % 2 == 0) ? '#F9F9F9' : '#FFFFFF',
                    color: 'black', textDecoration: ''
                };
            }
        }
    } 
}""".strip()

COLUMN_ROW_RENDERER_SCRATCH = """
function(params) {
    let row = helpers.convRow(params.node.id);
    let col = helpers.convCol(params.colDef.field);
    if (helpers.isSelected(row, col)) {
        return {backgroundColor: '#999999'};
    } else {
        return {backgroundColor: (parseInt(row) % 2 == 1) ? '#F9F9F9' : '#FFFFFF'};
    }
}""".strip()

COLUMN_ROW_RENDERER_SOLUTION = """
function(params) {
    let row = helpers.convRow(params.node.id);
    let col = helpers.convCol(params.colDef.field);
    let textDecoration = '';
    let isSelected = helpers.isSelected(row, col);

    if (params.node.id === "0") {
        if (isSelected) {
            return {backgroundColor: 'darkgreen', color: 'white', textDecoration: ''};
        } else {
            return {backgroundColor: '#008001', color: 'white', textDecoration: ''};
        }
    } else {
        if (isSelected) {
            return {backgroundColor: '#999999', color: 'black', textDecoration: ''};
        } else {
            return {
                backgroundColor: (parseInt(row) % 2 == 0) ? '#F9F9F9' : '#FFFFFF',
                color: 'black', textDecoration: ''
            };
        }
    } 
}""".strip()

ON_RANGE_SELECTION_CHANGED_NON_OUTPUT = """
function (eventData) {
    if (eventData.finished) { 
        let api_map = window._gauss_state[helpers.session].api_map;
        let out_api = api_map[window._gauss_state[helpers.session].out_api];
        out_api.helpers.clearSelected();
        helpers.updateSelected();
    }
}
"""

ON_RANGE_SELECTION_CHANGED_OUTPUT = """
function (eventData) {
    if (eventData.finished) {
        let api_map = window._gauss_state[helpers.session].api_map;
        window._gauss_state[helpers.session].non_output_apis.forEach(grid_id => {
            let api = api_map[grid_id];
            api.helpers.clearSelected();
        });
        
        helpers.updateSelected();
    }
}
"""


@attr.s(cmp=False, repr=False)
class GridWidget(ABC):
    session: UISession = attr.ib()
    grid_id: str = attr.ib()

    _widget = attr.ib(init=False)

    def __attrs_post_init__(self):
        grid_data = self.get_grid_data()

        self._widget = Grid(
            grid_data=grid_data,
            **self.get_grid_kwargs()
        )

    @abstractmethod
    def get_grid_data(self) -> pd.DataFrame:
        pass

    def get_widget(self):
        return self._widget

    def get_grid_kwargs(self):
        return {
            'theme': 'ag-theme-fresh',
            'columns_fit': 'auto',
            'grid_options': self.get_grid_options(),
            'js_helpers_custom': self.get_js_helpers(),
            'js_pre_grid': self.get_js_pre_grid(),
            'js_post_grid': self.get_js_post_grid(),
        }

    def get_grid_options(self):
        return {
            'enableBrowserTooltips': True,
            'headerHeight': 0,
            'enableRangeSelection': True,
            'domLayout': 'autoHeight',
            'suppressColumnVirtualisation': True,
            'onFirstDataRendered': """function () { 
                let parDiv = helpers.findOutputDiv();
                if (parDiv != undefined) {
                    helpers.api.setPopupParent(helpers.findOutputDiv());
                    helpers.popupParentSet = true;
                }
            }""",
            'onCellContextMenu': """function () {
                if (!helpers.popupParentSet) { 
                    helpers.api.setPopupParent(helpers.findOutputDiv());
                    helpers.popupParentSet = true;
                }
            }""",
        }

    def get_js_helpers(self):
        return gen_js_helpers(self.session.session_id, self.grid_id)

    def get_js_pre_grid(self):
        global_map_init = """
{
    if (window._gauss_state === undefined) {
        window._gauss_state = {};
    }
    
    if (window._gauss_state[SESSION_ID] === undefined) {
        window._gauss_state[SESSION_ID] = {};
        window._gauss_state[SESSION_ID].api_map = {};
        window._gauss_state[SESSION_ID].all_apis = new Set();
        window._gauss_state[SESSION_ID].inp_apis = new Set();
        window._gauss_state[SESSION_ID].non_output_apis = new Set();
        window._gauss_state[SESSION_ID].out_api = null;
        window._gauss_state[SESSION_ID].scratch_api = null;
        window._gauss_state[SESSION_ID].history_api = null;
        window._gauss_state[SESSION_ID].value_trackers = {};
        window._gauss_state[SESSION_ID].timestamp = 0;
    }
    
    if (window._gauss_state[KEY] === undefined) {
        window._gauss_state[KEY] = {};
    }
    
    window._gauss_state[KEY].selectedCells = new Set();
    window._gauss_state[KEY].orderedSelectedCells = [];
    window._gauss_state[KEY].markedForDeletion = new Set();
}
    
        """.strip().replace("SESSION_ID", repr(self.session.session_id)).replace("KEY",
                                                                                 repr(f"{self.session.session_id}:{self.grid_id}"))

        return [
            # Init global state
            global_map_init,
        ]

    def get_js_post_grid(self):
        common = [
            #  Save the session.
            f"helpers.session = {self.session.session_id!r};"
            f"helpers.grid_id = {self.grid_id!r};"
            #  Save the APIs
            "helpers.gridOptions = view.gridOptions;",
            "helpers.api = view.gridOptions.api;",
            "helpers.gridDiv = view.gridDiv;",
            "helpers.columnApi = view.gridOptions.columnApi;"
            "helpers.api.columnApi = view.gridOptions.columnApi;"
            "helpers.api.gridDiv = view.gridDiv;"
            #  Match height of the div to the grid.
            "view.gridDiv.style.height = 'auto';",
            #  Add the ID of the grid.
            f"view.gridOptions.api.gauss_grid_id = {self.grid_id!r};",
            #  Add in the helper as well.
            f"view.gridOptions.api.helpers = helpers;",
            #  Update global state.
            f"window._gauss_state[{self.session.session_id!r}].all_apis.add({self.grid_id!r});",
            f"window._gauss_state[{self.session.session_id!r}].api_map[{self.grid_id!r}] = view.gridOptions.api;",
            #  Setup popup parent.
            f"helpers.popupParentSet = false;",
            #  Notification Library
            NOTIFY_JS,
            #  General info.
            f"helpers.isScratch = false;",
        ]

        return common


@attr.s(cmp=False, repr=False)
class InputGridWidget(GridWidget):
    df: pd.DataFrame = attr.ib()

    _value_trackers: Dict[str, str] = attr.ib()

    def get_grid_data(self) -> pd.DataFrame:
        df = self.df
        if df.columns.nlevels > 1 or df.index.nlevels > 1:
            raise AssertionError("Multi-index dataframes not supported yet.")

        col_map = {c: f"C{idx}" for idx, c in enumerate(df.columns)}
        renamed_obj = df.rename(columns=col_map)
        records = [{v: k for k, v in col_map.items()}]
        records.extend(renamed_obj.to_dict('records'))

        return pd.DataFrame(records)

    def get_grid_options(self):
        return {
            **super().get_grid_options(),
            'defaultColDef': {
                'editable': False,
                'resizable': True,
                'cellStyle': COLUMN_ROW_RENDERER
            },
            'processCellForClipboard': genScratchClipboardCallbackForCell(self.session.session_id, self.grid_id),
            'getContextMenuItems': genContextMenuNonOutput(self.session),
            'onRangeSelectionChanged': ON_RANGE_SELECTION_CHANGED_NON_OUTPUT,
        }

    def get_js_post_grid(self):
        assignments = [f"trackers[{k!r}] = {v!r};"
                       for k, v in self._value_trackers.items()]

        value_tracking_info = """
{
    let trackers = window._gauss_state[SESSION_ID].value_trackers;
    ASSIGNMENTS
}
        """.strip().replace("SESSION_ID", repr(self.session.session_id)).replace("ASSIGNMENTS", "\n".join(assignments))

        return [
            *super().get_js_post_grid(),
            #  Update global state.
            f"console.log('Executing');",
            f"window._gauss_state[{self.session.session_id!r}].inp_apis.add({self.grid_id!r});",
            f"window._gauss_state[{self.session.session_id!r}].non_output_apis.add({self.grid_id!r});",
            #  Add value tracking information.
            value_tracking_info,
        ]


@attr.s(cmp=False, repr=False)
class OutputGridWidget(GridWidget):
    def get_grid_data(self) -> pd.DataFrame:
        num_rows = 3
        num_cols = 3
        placeholder = pd.DataFrame([['' for _ in range(num_cols)] for _ in range(num_rows)],
                                   columns=[f"C{i}" for i in range(num_cols)])

        return placeholder

    def get_grid_kwargs(self):
        return {
            **super().get_grid_kwargs(),
            'menu': {
                'buttons': [
                    {'name': 'Add Column', 'action': "helpers.addColumn();"},
                    {'name': 'Add Row', 'action': "helpers.addRow();"},
                ],
            },
        }

    def get_grid_options(self):
        return {
            **super().get_grid_options(),
            'defaultColDef': {
                'editable': True,
                'resizable': True,
                'cellStyle': COLUMN_ROW_RENDERER_OUTPUT
            },
            'suppressContextMenu': True,
            'processCellFromClipboard': genOutputClipboardCallbackCell(self.session.session_id, self.grid_id),
            'processDataFromClipboard': genOutputClipboardCallbackData(self.session.session_id, self.grid_id),
            'getContextMenuItems': genContextMenuOutput(self.session),
            'onRangeSelectionChanged': ON_RANGE_SELECTION_CHANGED_OUTPUT,
            'suppressClickEdit': True,
            'onPasteEnd': """function () { helpers.gridOptions.columnApi.autoSizeAllColumns(); }""",
            "suppressKeyboardEvent": """function (params) {
                let KEY_V = 86;
                let KEY_DELETE = 46;
                let KEY_BACKSPACE = 8;
                let event = params.event;
                let key = event.which;
                if (event.ctrlKey || event.metaKey) {
                    if (key == KEY_V) return false;
                } else if (key == KEY_DELETE || key == KEY_BACKSPACE) {
                    if (helpers.getNumSelectedCells() > 0) {
                        let selectedCells = helpers.getSelectedCells();
                        selectedCells.forEach(cell => {
                            let splitted = cell.split(':');
                            let row = splitted[0];
                            let col = splitted[1];
                            helpers.clearCell(row, col);
                        });
                        
                        let session = helpers.session;
                        let args = JSON.stringify(selectedCells);
                        let cmd = "gauss.ui.get_session(\\"" + session + "\\")._clear_output_cells(" + args + ")";
                        IPython.notebook.kernel.execute(cmd);
                    }
                }
                
                return true;
            }""".strip()
        }

    def get_js_post_grid(self):
        return [
            *super().get_js_post_grid(),
            #  Update global state.
            f"window._gauss_state[{self.session.session_id!r}].out_api = {self.grid_id!r};",
        ]


@attr.s(cmp=False, repr=False)
class ScratchGridWidget(GridWidget):
    def get_grid_data(self) -> pd.DataFrame:
        num_rows = 1
        num_cols = 1
        placeholder = pd.DataFrame([['' for _ in range(num_cols)] for _ in range(num_rows)],
                                   columns=[f"C{i}" for i in range(num_cols)])

        return placeholder

    def get_grid_kwargs(self):
        return {
            **super().get_grid_kwargs(),
            'columns_fit': 'size_to_fit',
            'menu': {
                'buttons': [
                    {'name': 'Add Column', 'action': "helpers.addColumn();"},
                    {'name': 'Add Row', 'action': "helpers.addRow();"},
                ],
            },
        }

    def get_grid_options(self):
        return {
            **super().get_grid_options(),
            'defaultColDef': {
                'editable': True,
                'resizable': True,
                'cellStyle': COLUMN_ROW_RENDERER_SCRATCH,
            },
            'processCellForClipboard': genScratchClipboardCallbackForCell(self.session.session_id, self.grid_id),
            'processCellFromClipboard': genScratchClipboardCallbackFromCell(self.session.session_id, self.grid_id),
            'processDataFromClipboard': genScratchClipboardCallbackFromData(self.session.session_id, self.grid_id),
            'getContextMenuItems': genContextMenuNonOutput(self.session),
            'onRangeSelectionChanged': ON_RANGE_SELECTION_CHANGED_NON_OUTPUT,
        }

    def get_js_post_grid(self):
        return [
            *super().get_js_post_grid(),
            #  Update global state.
            f"window._gauss_state[{self.session.session_id!r}].scratch_api = {self.grid_id!r};",
            f"window._gauss_state[{self.session.session_id!r}].non_output_apis.add({self.grid_id!r});",
            f"helpers.isScratch = true;",
        ]


@attr.s(cmp=False, repr=False)
class SolutionGridWidget(GridWidget):
    df: pd.DataFrame = attr.ib(default=None)
    explanations: Dict[Tuple[int, int], Tuple[List[Tuple[int, int, int]], Optional[str]]] = attr.ib(default=None)

    def get_grid_data(self) -> pd.DataFrame:
        if self.df is not None:
            df = self.df
            if df.columns.nlevels > 1 or df.index.nlevels > 1:
                raise AssertionError("Multi-index dataframes not supported yet.")

            col_map = {c: f"C{idx}" for idx, c in enumerate(df.columns)}
            renamed_obj = df.rename(columns=col_map)
            records = [{v: k for k, v in col_map.items()}]
            records.extend(renamed_obj.to_dict('records'))

            return pd.DataFrame(records)

        num_rows = 3
        num_cols = 3
        placeholder = pd.DataFrame([['' for _ in range(num_cols)] for _ in range(num_rows)],
                                   columns=[f"C{i}" for i in range(num_cols)])

        return placeholder

    def get_grid_options(self):
        return {
            'enableBrowserTooltips': True,
            'headerHeight': 0,
            'domLayout': 'autoHeight',
            'suppressColumnVirtualisation': True,
            'defaultColDef': {
                'editable': False,
                'resizable': True,
                'cellStyle': COLUMN_ROW_RENDERER_SOLUTION
            },
            'suppressContextMenu': True,
            'enableRangeSelection': False,
            'onCellClicked': """function (e) { 
            
            let row = helpers.convRow(e.node.id);
            let col = helpers.convCol(e.column.colDef.field);
            let cmd = "gauss.ui.get_session(\\"" + helpers.session + "\\").explain(" + row + ", " + col + ")";
            console.log(cmd);
            IPython.notebook.kernel.execute(cmd);
            }""".strip()
        }

    def update(self, df: pd.DataFrame, explanations):
        self.explanations = explanations
        new_col_defs = [{"field": f"C{idx}", "headerName": f"C{idx}"} for idx, col in enumerate(df.columns)]
        new_data = [{f"C{idx}": col for idx, col in enumerate(df.columns)}]

        for row in df.iterrows():
            row = row[1]
            new_data.append({f"C{idx}": str(val) for idx, val in enumerate(row)})

        code = Javascript("""{
        let api_map = window._gauss_state["SESSION_ID"].api_map;
        let api = api_map["GRID_ID"];
        api.setColumnDefs(COL_DEFS);
        api.setRowData(ROW_DEFS);
        api.columnApi.autoSizeAllColumns();
}""".replace("SESSION_ID", self.session.session_id).replace("GRID_ID", self.grid_id)
                           .replace("COL_DEFS", repr(new_col_defs)).replace("ROW_DEFS", repr(new_data)))
        display(code)

    def get_explanation(self, row_id: int, col_id: int) -> str:
        involved_inp_nodes, expr_str = self.explanations[row_id, col_id]
        involved_cells = [[f"I{inp_id}", str(row + 1), f"C{col}"] for inp_id, row, col in involved_inp_nodes]

        code = Javascript("""{
                let api_map = window._gauss_state["SESSION_ID"].api_map;
                INVOLVED.forEach((entry) => {
                    let grid_id = entry[0];
                    let row = entry[1];
                    let col = entry[2];
                    let api = api_map[grid_id];
                    api.flashCells({rowNodes: [api.getRowNode(row)], columns: [col], flashDelay: 8000}); 
                })
        }""".replace("SESSION_ID", self.session.session_id).replace("INVOLVED", repr(involved_cells)))
        display(code)

        if expr_str is not None and expr_str.startswith("(") and expr_str.endswith(")"):
            expr_str = expr_str[1:-1]

        return f"={expr_str}" if expr_str is not None else None

