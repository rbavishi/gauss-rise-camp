from gauss.ui.grids.common import GAUSS_MAGIC_PREFIX
from gauss.ui.grids.operations import getContextMenuOpItems
from gauss.ui.session import UISession


def genInputClipboardCallbackForCell(session_id: str, g_id: str):
    return """
function processCopyToClipboard(params) {
    let rowNum = helpers.convRow(params.node.id);
    let colNum = helpers.convCol(params.column.colId); 
    let key = rowNum + ":" + colNum + ":" + helpers.api.gauss_grid_id;
    let tracker = window._gauss_state[helpers.session].value_trackers[key];
    return ("GAUSS_MAGIC_PREFIX" + tracker + "GAUSS_MAGIC_PREFIX" + params.value);
}
""".strip().replace("GAUSS_MAGIC_PREFIX", GAUSS_MAGIC_PREFIX)


def genScratchClipboardCallbackForCell(session_id: str, g_id: str):
    return """
function processCopyToClipboard(params) {
    let rowNum = helpers.convRow(params.node.id);
    let colNum = helpers.convCol(params.column.colId); 
    let key = rowNum + ":" + colNum + ":" + helpers.api.gauss_grid_id;
    let tracker = window._gauss_state[helpers.session].value_trackers[key];
    return ("GAUSS_MAGIC_PREFIX" + tracker + "GAUSS_MAGIC_PREFIX" + params.value);
}
""".strip().replace("GAUSS_MAGIC_PREFIX", GAUSS_MAGIC_PREFIX)


def genScratchClipboardCallbackFromData(session_id: str, g_id: str):
    return """
function (params) {
    let selected = helpers.getSelectedCells();
    if (params.data.length <= 0) {
        return null;
    }
    
    let numDataElems = params.data.length * params.data[0].length;
    
    if (numDataElems !== selected.length) {
        if (selected.length === 1) {
            let s_item = selected[0].split(":");
            let s_row = parseInt(s_item[0]);
            let s_col = parseInt(s_item[1]);
            let g_id = s_item[2];
            
            let numRows = helpers.getNumRows();
            let numColumns = helpers.getNumColumns();
            
            let requiredRows = params.data.length + s_row;
            let requiredColumns = params.data[0].length + s_col;
            while (requiredRows > numRows) {
                requiredRows -=1;
                helpers.addRow();
            }
            
            while (requiredColumns > numColumns) {
                requiredColumns -= 1;
                helpers.addColumn();
            }
            
            selected = [];
            for (let r = 0; r < params.data.length; r++) {
                for (let c = 0; c < params.data[0].length; c++) {
                    selected.push((r + s_row) + ":" + (c + s_col) + ":" + g_id);
                }
            }
            
        } else {
            $.notify("Error : Select " + numDataElems + " cells to copy into.", 
                     $.notify.defaults({className: "error", "position": "left bottom"}));
            return null;
        }
    }
    
    let minR = null;
    let maxR = null;
    let minC = null;
    let maxC = null;
    selected.forEach(item => {
        let s_item = item.split(":");
        let curR = parseInt(s_item[0]);
        let curC = parseInt(s_item[1]);
        if (minR == null || minR > curR) minR = curR;
        if (maxR == null || maxR < curR) maxR = curR;
        if (minC == null || minC > curC) minC = curC;
        if (maxC == null || maxC < curC) maxC = curC;
    });
    
    let newData = [];
    for (let r = 0; r < (maxR - minR + 1); r++) {
        let row = [];
        newData.push(row);
        for (let c = 0; c < (maxC - minC + 1); c++) {
            row.push("GAUSS_MAGIC_PREFIXGAUSS_MAGIC_PREFIX");
        }
    }
    
    let index = 0;
    for (let r = 0; r < params.data.length; r++) {
        for (let c = 0; c < params.data[0].length; c++) {
            let item = selected[index].split(":");
            index += 1;
            let curR = parseInt(item[0]);
            let curC = parseInt(item[1]);
            newData[curR - minR][curC - minC] = params.data[r][c];
        }
    }
    
    helpers.api.clearRangeSelection();
    helpers.api.addCellRange({
        rowStartIndex: parseInt(helpers.invRow(minR)),
        rowEndIndex: parseInt(helpers.invRow(maxR)),
        columnStart: helpers.invCol(minC),
        columnEnd: helpers.invCol(maxC),
    });
    
    console.log(newData);
    
    return newData;
}
""".strip().replace("GAUSS_MAGIC_PREFIX", GAUSS_MAGIC_PREFIX)


def genScratchClipboardCallbackFromCell(session_id: str, g_id: str):
    return """
function (params) {
    console.log("HELLO");
    console.log(params);
    if (params.value.startsWith("GAUSS_MAGIC_PREFIX")) {
        let splitted = params.value.split("GAUSS_MAGIC_PREFIX");
        let tracker = splitted[1];
        if (tracker === "") {
            //  Deliberately empty, no change.
            return null;
        } 
         
        let rowNum = helpers.convRow(params.node.id);
        let colNum = helpers.convCol(params.column.colId); 
        let key = rowNum + ":" + colNum + ":" + helpers.api.gauss_grid_id;
        
        window._gauss_state[helpers.session].value_trackers[key] = tracker;
        return splitted[splitted.length - 1];
        
    } else {
        $.notify("Can only copy values from input(s) or results of computations", 
                 $.notify.defaults({className: "error", "position": "left bottom"}))
    }
}
""".strip().replace("GAUSS_MAGIC_PREFIX", GAUSS_MAGIC_PREFIX)


def genOutputClipboardCallbackData(session_id: str, g_id: str):
    return """
function (params) {
    let selected = helpers.getSelectedCells();
    if (params.data.length <= 0) {
        return null;
    }
    
    let numDataElems = params.data.length * params.data[0].length;
    
    if (numDataElems !== selected.length) {
        if (selected.length === 1) {
            let s_item = selected[0].split(":");
            let s_row = parseInt(s_item[0]);
            let s_col = parseInt(s_item[1]);
            let g_id = s_item[2];
            
            let numRows = helpers.getNumRows();
            let numColumns = helpers.getNumColumns();
            
            let requiredRows = params.data.length + s_row;
            let requiredColumns = params.data[0].length + s_col;
            while (requiredRows > numRows) {
                requiredRows -=1;
                helpers.addRow();
            }
            
            while (requiredColumns > numColumns) {
                requiredColumns -= 1;
                helpers.addColumn();
            }
            
            selected = [];
            for (let r = 0; r < params.data.length; r++) {
                for (let c = 0; c < params.data[0].length; c++) {
                    selected.push((r + s_row) + ":" + (c + s_col) + ":" + g_id);
                }
            }
            
        } else {
            $.notify("Error : Select " + numDataElems + " cells to copy into.", 
                     $.notify.defaults({className: "error", "position": "left bottom"}));
            return null;
        }
        
    }
    
    let minR = null;
    let maxR = null;
    let minC = null;
    let maxC = null;
    selected.forEach(item => {
        let s_item = item.split(":");
        let curR = parseInt(s_item[0]);
        let curC = parseInt(s_item[1]);
        if (minR == null || minR > curR) minR = curR;
        if (maxR == null || maxR < curR) maxR = curR;
        if (minC == null || minC > curC) minC = curC;
        if (maxC == null || maxC < curC) maxC = curC;
    });
    
    let newData = [];
    for (let r = 0; r < (maxR - minR + 1); r++) {
        let row = [];
        newData.push(row);
        for (let c = 0; c < (maxC - minC + 1); c++) {
            row.push("GAUSS_MAGIC_PREFIXGAUSS_MAGIC_PREFIX");
        }
    }
    
    let index = 0;
    for (let r = 0; r < params.data.length; r++) {
        for (let c = 0; c < params.data[0].length; c++) {
            let item = selected[index].split(":");
            index += 1;
            let curR = parseInt(item[0]);
            let curC = parseInt(item[1]);
            newData[curR - minR][curC - minC] = params.data[r][c];
        }
    }
    
    helpers.api.clearRangeSelection();
    helpers.api.addCellRange({
        rowStartIndex: parseInt(helpers.invRow(minR)),
        rowEndIndex: parseInt(helpers.invRow(maxR)),
        columnStart: helpers.invCol(minC),
        columnEnd: helpers.invCol(maxC),
    });
    
    return newData;
}
""".strip().replace("GAUSS_MAGIC_PREFIX", GAUSS_MAGIC_PREFIX)


def genOutputClipboardCallbackCell(session_id: str, g_id: str):
    return """
function (params) {
    if (params.value.startsWith("GAUSS_MAGIC_PREFIX")) {
        let splitted = params.value.split("GAUSS_MAGIC_PREFIX");
        let tracker = splitted[1];
        if (tracker === "") {
            //  Deliberately empty, no change.
            return helpers.api.getValue(params.column.colId, params.node);
        } 
        
        let rowNum = helpers.convRow(params.node.id);
        let colNum = helpers.convCol(params.column.colId); 
        let interaction_args = "\\"" + tracker + "\\", \\"" + rowNum + "\\", \\"" + colNum + "\\"";
        let cmd = "gauss.ui.get_session(\\"" + helpers.session + "\\").record_interaction(" + interaction_args + ")";
        console.log(cmd);
        IPython.notebook.kernel.execute(cmd);
        return splitted[splitted.length - 1];
         
    } else {
        $.notify("Can only copy values from input(s) or results of computations", 
                 $.notify.defaults({className: "error", "position": "left bottom"}));
                 
        return helpers.api.getValue(params.column.colId, params.node);
    }
}
""".strip().replace("GAUSS_MAGIC_PREFIX", GAUSS_MAGIC_PREFIX)


def genContextMenuNonOutput(session: UISession):
    return """
function (params) {
    let selectedCells = helpers.getSelectedCellsNonOutput();
    let numSelectedCells = selectedCells.length;
    let isSingleColumnSelected = (numSelectedCells === 1) && (selectedCells[0].split(':')[0] < 0); 
    let isNonInputSelected = (numSelectedCells == 0) || selectedCells.some(e => (!e.split(':')[2].startsWith("I")));
    
    let result = [
        'copy',
        'paste',
        'autoSizeAll',
        'separator',
        MENU_ITEMS
    ];
    
    return result;
}
""".strip().replace("MENU_ITEMS", getContextMenuOpItems(session.domain_ui.get_available_operations()))


def genContextMenuOutput(session: UISession):
    return """
function (params) {
    let result = [
        'paste',
        'autoSizeAll',
    ];
    
    return result;
}
"""
