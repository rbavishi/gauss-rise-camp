def genConvRow(session_id: str, grid_id: str):
    return """
function (rowId) {
    if (helpers.isScratch) return rowId;
    if (rowId === "0") {
        return "-1";
    } else {
        return (parseInt(rowId) - 1).toString();
    }
}""".strip()


def genInvRow(session_id: str, grid_id: str):
    return """
function (rowId) {
    if (helpers.isScratch) return rowId;
    if (rowId === "-1") {
        return "0";
    } else {
        return (parseInt(rowId) + 1).toString();
    }
} 
""".strip()


def genConvCol(session_id: str, grid_id: str):
    return """
function (colId) {
    return colId.substr(1);
}""".strip()


def genInvCol(session_id: str, grid_id: str):
    return """
function (colId, api) {
    return "C" + colId;
}
""".strip()


def genAddColumn(session_id: str, grid_id: str):
    return """
function() {
    var columnDefs = helpers.gridOptions.columnDefs;
    var newCol = "C" + columnDefs.length;
    columnDefs.push({ field:newCol, headerName: newCol});
    helpers.gridOptions.api.setColumnDefs(columnDefs);
    helpers.gridOptions.columnApi.autoSizeAllColumns();
}
    """.strip()


def genAddRow(session_id: str, grid_id: str):
    return """
function () {
    let record = {};
    helpers.gridOptions.columnDefs.forEach(c => {record[c.field] = '';});
    helpers.gridOptions.api.updateRowData({add: [record]});
    helpers.gridOptions.columnApi.autoSizeAllColumns();
}
    """.strip()


def genGetNumRows(session_id: str, grid_id: str):
    return """
function () {
    let num = 0;
    helpers.api.forEachNode(function (rowNode, index) { num += 1; });
    if (helpers.isScratch) return num;
    return num - 1;
}
    """.strip()


def genGetNumColumns(session_id: str, grid_id: str):
    return """
function () {
    return helpers.gridOptions.columnDefs.length;
}
    """.strip()


def genGetRowDiv(session_id: str, grid_id: str):
    return """
function (row_id) {
    let key = "SESSION_ID:GRID_ID:" + row_id;
    console.log(key);
    console.log('[row-business-key="' + key + '"]');
    let results = $('[row-business-key="' + key + '"]');
    console.log(results);
    for (let i = 0; i < results.length; i++) {
        if ($(results[i]).is(":visible"))
         return results[i];
    }
}
""".strip().replace("SESSION_ID", session_id).replace("GRID_ID", grid_id)


def genGetCellDiv(session_id: str, grid_id: str):
    return """
function (row_id, col_id) {
    let rowDiv = helpers.getRowDiv(row_id);
    return $(rowDiv).children()[col_id];
}   
""".strip()


def genEnableEditing(session_id: str, grid_id: str):
    return """
function () {
    console.log("Did");
    helpers.gridOptions.columnDefs.forEach(c => { console.log(c); c.editable = true; });
    helpers.gridOptions.api.setColumnDefs(helpers.gridOptions.columnDefs);
}
    """.strip()


def genGetAllSelectedCells(session_id: str, grid_id: str):
    return """
function () {
    let selected = [];
    let apis = window._gauss_state[helpers.session].all_apis;
    apis.forEach((api) => {
        let key = helpers.session + ":" + api.gauss_grid_id;
        let orderedSelectedCells = window._gauss_state[key].orderedSelectedCells;
        orderedSelectedCells.forEach(e => { selected.push(e); }); 
    });

    selected.sort(function(first, second) {
        return first[1] - second[1];
    });
    
    return selected.map(e => e[0]);
}
""".strip()


def genGetSelectedCellsNonOutput(session_id: str, grid_id: str):
    return """
function () {
    let selected = [];
    let api_map = window._gauss_state[helpers.session].api_map;
    let grid_ids = window._gauss_state[helpers.session].non_output_apis;
    grid_ids.forEach(grid_id => { 
        let api = api_map[grid_id];
        let key = helpers.session + ":" + api.gauss_grid_id;
        let orderedSelectedCells = window._gauss_state[key].orderedSelectedCells;
        orderedSelectedCells.forEach(e => { selected.push(e); }); 
    });

    selected.sort(function(first, second) {
        return first[1] - second[1];
    });
    
    return selected.map(e => e[0]);
}
""".strip()


def genGetNumSelectedCellsNonOutput(session_id: str, grid_id: str):
    return """
function () {
    let api_map = window._gauss_state[helpers.session].api_map;
    let grid_ids = window._gauss_state[helpers.session].non_output_apis;
    let num = 0;
    grid_ids.forEach(grid_id => { 
        let api = api_map[grid_id];
        let key = helpers.session + ":" + api.gauss_grid_id;
        let orderedSelectedCells = window._gauss_state[key].orderedSelectedCells;
        num += orderedSelectedCells.length;
    });
    
    return num;
}
""".strip()


def genGetSelectedCells(session_id: str, grid_id: str):
    return """
function () {
    let selected = [];
    let api = helpers.api;
    let key = helpers.session + ":" + api.gauss_grid_id;
    let orderedSelectedCells = window._gauss_state[key].orderedSelectedCells;
    orderedSelectedCells.forEach(e => { selected.push(e); });

    selected.sort(function(first, second) {
        return first[1] - second[1];
    });
    
    return selected.map(e => e[0]);
}
""".strip()


def genGetNumSelectedCells(session_id: str, grid_id: str):
    return """
function () {
    let selected = [];
    let api = helpers.api;
    let key = helpers.session + ":" + api.gauss_grid_id;
    let orderedSelectedCells = window._gauss_state[key].orderedSelectedCells;
    return orderedSelectedCells.length;
}
""".strip()


def genIsSelected(session_id: str, grid_id: str):
    key_set = f"{session_id}:{grid_id}"
    return """
function (row, col) { 
    if (window._gauss_state === undefined) return false;
    if (window._gauss_state[KEY] === undefined) return false;
    
    let key = row + ":" + col;
    return (window._gauss_state[KEY].selectedCells.has(key)); 
}
""".strip().replace("KEY", repr(key_set))


def genIsMarkedForDeletion(session_id: str, grid_id: str):
    key_set = f"{session_id}:{grid_id}"
    return """
function (row, col) { 
    if (window._gauss_state === undefined) return false;
    if (window._gauss_state[KEY] === undefined) return false;
    
    let key = row + ":" + col;
    return (window._gauss_state[KEY].markedForDeletion.has(key)); 
}
""".strip().replace("KEY", repr(key_set))


def genUpdateSelected(session_id: str, grid_id: str):
    key_set = f"{session_id}:{grid_id}"
    return """
function () {
    let selected = [];
    let api = helpers.api;
    let ranges = api.getCellRanges();
    let oldSelectedCells = window._gauss_state[KEY].selectedCells;
    let newSelectedCells = new Set();
    let newSelectedCellsArray = [];
    let orderedSelectedCells = [];
    let changed = [];
    ranges.forEach((r) => {
        let rStart = r.startRow.rowIndex;
        let rEnd = r.endRow.rowIndex;
        if (rStart > rEnd) {
            let tmp = rStart;
            rStart = rEnd;
            rEnd = tmp;
        }
        
        for (let row = rStart; row <= rEnd; row++) {
            r.columns.forEach((c) => {
                let key = helpers.convRow(row.toString()) + ":" + helpers.convCol(c.colId);
                if (!newSelectedCells.has(key)) {
                    newSelectedCellsArray.push(key);
                }
                newSelectedCells.add(key);
                if (!oldSelectedCells.has(key)) {
                    changed.push(key);
                }
            });
        };
    });
    
    if (oldSelectedCells.size === 1 && newSelectedCells.size === 1 && changed.length === 0) {
        changed.push(oldSelectedCells.values().next().value);
        newSelectedCells.clear();
        window._gauss_state[KEY].selectedCells = newSelectedCells;
        api.deselectAll();
        api.clearRangeSelection();
        
    } else {
        window._gauss_state[KEY].selectedCells = newSelectedCells;
        
        oldSelectedCells.forEach(c => {
            if (!newSelectedCells.has(c)) {
                changed.push(c);
            }
        });
    }

    changed.forEach(c => {
        let splitted = c.split(':');
        let row = api.getRowNode(helpers.invRow(splitted[0]));
        let col = helpers.invCol(splitted[1]);
        api.refreshCells({rowNodes: [row], columns: [col], force: true});
    });
    
    newSelectedCellsArray.forEach(c => {
        helpers.updateTimestamp();
        orderedSelectedCells.push([c + ":" + api.gauss_grid_id, helpers.getTimestamp()]);
    });
    
    window._gauss_state[KEY].orderedSelectedCells = orderedSelectedCells;
} 
""".strip().replace("KEY", repr(key_set))


def genClearSelected(session_id: str, grid_id: str):
    key_set = f"{session_id}:{grid_id}"
    return """
    function () {
        let selected = [];
        let api = helpers.api;
        let oldSelectedCells = window._gauss_state[KEY].selectedCells;
        if (oldSelectedCells.size === 0) return;
        
        let newSelectedCells = new Set();
        let orderedSelectedCells = [];
        let changed = [];
        
        if (oldSelectedCells.size === 1 && newSelectedCells.size === 1 && changed.length === 0) {
            changed.push(oldSelectedCells.values().next().value);
            newSelectedCells.clear();
            window._gauss_state[KEY].selectedCells = newSelectedCells;
            api.deselectAll();
            api.clearRangeSelection();
            
        } else {
            window._gauss_state[KEY].selectedCells = newSelectedCells;
            
            oldSelectedCells.forEach(c => {
                if (!newSelectedCells.has(c)) {
                    changed.push(c);
                }
            });
        }

        changed.forEach(c => {
            let splitted = c.split(':');
            let row = api.getRowNode(helpers.invRow(splitted[0]));
            let col = helpers.invCol(splitted[1]);
            api.refreshCells({rowNodes: [row], columns: [col], force: true});
        });
        
        newSelectedCells.forEach(c => {
            helpers.updateTimestamp();
            orderedSelectedCells.push([c + ":" + api.gauss_grid_id, helpers.getTimestamp()]);
        });
        
        window._gauss_state[KEY].orderedSelectedCells = orderedSelectedCells;
    } 
    """.strip().replace("KEY", repr(key_set))


def genGetTime(session_id: str, grid_id: str):
    return """
function () {
    return window._gauss_state[SESSION].timestamp; 
}
""".strip().replace("SESSION", repr(session_id))
    pass


def genUpdateTime(session_id: str, grid_id: str):
    return """
function () { 
    window._gauss_state[SESSION].timestamp += 1;
}
""".strip().replace("SESSION", repr(session_id))


COPY_TO_CLIPBOARD = """
function (str) {
    const el = document.createElement('textarea');
    el.value = str;
    el.setAttribute('readonly', '');
    el.style.position = 'absolute';
    el.style.left = '-9999px';
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
}
""".strip()


FIND_OUTPUT_DIV = """
function () {
    let cur = helpers.gridDiv.parentNode;
    console.log(helpers.gridDiv.parentElement);
    while (!(cur.className === "output" && cur.parentNode.className === "output_wrapper")) {
        cur = cur.parentNode;
    }
    
    return cur;
}
""".strip()

CLEAR_CELL = """
function (row, col) {
    let KEY_DELETE = 46;
    helpers.api.startEditingCell({
        rowIndex: parseInt(helpers.invRow(row)),
        colKey: helpers.invCol(col),
        keyPress: KEY_DELETE,
    });
    
    helpers.api.stopEditing(false);
}
"""


def gen_js_helpers(session_id: str, grid_id: str):
    return """
helpersCustom = {
    convRow: CONV_ROW,
    convCol: CONV_COL,
    invRow: INV_ROW,
    invCol: INV_COL,
    addRow: ADD_ROW,
    addColumn: ADD_COLUMN,
    getNumRows: GET_NUM_ROWS,
    getNumColumns: GET_NUM_COLUMNS,
    getRowDiv: GET_ROW_DIV,
    getCellDiv: GET_CELL_DIV,
    enableEditing: ENABLE_EDITING,
    getSelectedCells: GET_SELECTED_CELLS,
    getSelectedCellsNonOutput: GET_SELECTED_CELLS_NON_OUTPUT,
    getNumSelectedCellsNonOutput: GET_NUM_SELECTED_CELLS_NON_OUTPUT,
    getNumSelectedCells: GET_NUM_SELECTED_CELLS,
    clearCell: CLEAR_CELL,
    isSelected: IS_SELECTED,
    isMarkedForDeletion: IS_MARKED_FOR_DELETION,
    updateSelected: UPDATE_SELECTED,
    clearSelected: CLEAR_SELECTED,
    getTimestamp: GET_TIMESTAMP,
    updateTimestamp: UPDATE_TIMESTAMP,
    copyToClipboard: COPY_TO_CLIPBOARD,
    findOutputDiv: FIND_OUTPUT_DIV,
}
    """.strip().replace("CONV_ROW", genConvRow(session_id, grid_id)) \
        .replace("CONV_COL", genConvCol(session_id, grid_id)) \
        .replace("INV_ROW", genInvRow(session_id, grid_id)) \
        .replace("INV_COL", genInvCol(session_id, grid_id)) \
        .replace("ADD_ROW", genAddRow(session_id, grid_id)) \
        .replace("ADD_COLUMN", genAddColumn(session_id, grid_id)) \
        .replace("GET_NUM_ROWS", genGetNumRows(session_id, grid_id)) \
        .replace("GET_NUM_COLUMNS", genGetNumColumns(session_id, grid_id)) \
        .replace("GET_ROW_DIV", genGetRowDiv(session_id, grid_id)) \
        .replace("GET_CELL_DIV", genGetCellDiv(session_id, grid_id)) \
        .replace("ENABLE_EDITING", genEnableEditing(session_id, grid_id)) \
        .replace("GET_NUM_SELECTED_CELLS_NON_OUTPUT", genGetNumSelectedCellsNonOutput(session_id, grid_id)) \
        .replace("GET_NUM_SELECTED_CELLS", genGetNumSelectedCells(session_id, grid_id)) \
        .replace("GET_SELECTED_CELLS_NON_OUTPUT", genGetSelectedCellsNonOutput(session_id, grid_id)) \
        .replace("GET_SELECTED_CELLS", genGetSelectedCells(session_id, grid_id)) \
        .replace("CLEAR_CELL", CLEAR_CELL) \
        .replace("IS_SELECTED", genIsSelected(session_id, grid_id)) \
        .replace("IS_MARKED_FOR_DELETION", genIsMarkedForDeletion(session_id, grid_id)) \
        .replace("UPDATE_SELECTED", genUpdateSelected(session_id, grid_id)) \
        .replace("CLEAR_SELECTED", genClearSelected(session_id, grid_id)) \
        .replace("GET_TIMESTAMP", genGetTime(session_id, grid_id)) \
        .replace("UPDATE_TIMESTAMP", genUpdateTime(session_id, grid_id)) \
        .replace("COPY_TO_CLIPBOARD", COPY_TO_CLIPBOARD) \
        .replace("FIND_OUTPUT_DIV", FIND_OUTPUT_DIV)
