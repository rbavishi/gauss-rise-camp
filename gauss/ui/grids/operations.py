from typing import Dict, Any, List


class _EmbedString:
    def __init__(self, s: str):
        self.s = s

    def __repr__(self):
        return self.s


def getBuiltinAction(config: Dict[str, Any]) -> str:
    if config['builtin'] not in ['DELETE']:
        raise AssertionError(f"Builtin op not recognized {config['builtin']!r}")

    payload = {
        _EmbedString('cells'): _EmbedString('selected_cells'),
        _EmbedString('operation'): "DELETE"
    }

    return """
    function () {
        let selected_cells = Array.from(selectedCells);
        selectedCells.forEach(e => {
            let splitted = e.split(':');
            let grid_id = splitted[2];
            let r = splitted[0];
            let c = splitted[1];
            window._gauss_state[helpers.session + ":" + grid_id].markedForDeletion.add(r + ":" + c);
            let api = window._gauss_state[helpers.session].api_map[grid_id];
            let row = api.getRowNode(helpers.invRow(r));
            let col = api.helpers.invCol(c);
            api.refreshCells({rowNodes: [row], columns: [col], force: true});
        });
        
        let callbacks = {
            iopub: {
                output: (data) => {
                }
            }
        };
        
        let payload = __PAYLOAD_STR__;
        console.log(payload);
        let payload_str = JSON.stringify(payload);
        console.log(payload_str);
        let cmd = "gauss.ui.get_session(\\"" + helpers.session + "\\").process_computation(" + payload_str + ")";
        console.log(cmd);
        IPython.notebook.kernel.execute(cmd, callbacks);
    }
    """.strip().replace("__PAYLOAD_STR__", repr(payload))


def getOpAction(config: Dict[str, Any]) -> str:
    setup = []
    payload = {
        _EmbedString('cells'): _EmbedString('cells'),
        _EmbedString('operation'): _EmbedString("__OPERATION__")
    }

    if config.get('dialog', None) is not None:
        setup.append(f"let dialog = '{config['dialog']}';")
        payload[_EmbedString('user_input')] = _EmbedString(f"prompt(dialog)")

    return """
    function() {
                let selected_cells = helpers.getSelectedCellsNonOutput();
                let value_trackers = window._gauss_state[helpers.session].value_trackers;
                let cells = [];
                for (let i = 0; i < selected_cells.length; i++) {
                    let cell = selected_cells[i];
                    let tracker = value_trackers[cell];
                    if (tracker === undefined) {
                        $.notify("Error : Cannot use selected cells for computation", "error", 
                                 $.notify.defaults({className: "error", "position": "bottom center"}));
                        return;
                    }
                    
                    cells.push(tracker);
                }
                
                __SETUP__
                let payload = __PAYLOAD_STR__;
                if (payload.user_input !== undefined) {
                    payload.user_input = payload.user_input.replace(/'/g, "\\'");
                    console.log(payload.user_input);
                }
                
                let callbacks = {
                    iopub: {
                        output: (data) => {
                            console.log("Okay");
                            console.log(data.content);
                            console.log(data.content.text.trim());
                            result = JSON.parse(data.content.text.trim());
                            if (result.success === "true") {
                                helpers.copyToClipboard(result.value);
                                $.notify("Copied result to clipboard", 
                                         $.notify.defaults({className: "info", "position": "bottom center"}));
                            } else {
                                helpers.copyToClipboard("");
                                $.notify("Error : " + result.msg, "error", 
                                         $.notify.defaults({className: "error", "position": "bottom center"}));
                            }
                        }
                    }
                };
                
                let payload_str = JSON.stringify(payload);
                console.log(payload_str);
                let cmd = "gauss.ui.get_session(\\"" + helpers.session + "\\").process_computation(" + payload_str + ")";
                console.log(cmd);
                IPython.notebook.kernel.execute(cmd, callbacks);
            },
    """.strip() \
        .replace("__SETUP__", "\n".join(setup)) \
        .replace("__PAYLOAD_STR__", repr(payload)) \
        .replace("__OPERATION__", '"' + config['name'].upper() + '"')


def getContextMenuOpItems(config: List[Dict[str, Any]]) -> str:
    items = []
    for item_config in config:
        enabled_for = item_config.get('enabledFor', 'ALL')

        if enabled_for == 'ALL':
            disabled = "(numSelectedCells == 0)"
        elif enabled_for == 'SINGLE_COLUMN':
            disabled = "(!isSingleColumnSelected)"
        elif enabled_for == 'ONLY_INPUTS':
            disabled = "(isNonInputSelected)"
        else:
            raise ValueError(f"Incorrect value for 'enabledFor': {enabled_for}")

        if 'arity' in item_config:
            disabled = f"({_EmbedString(disabled)!r} || (numSelectedCells !== {item_config['arity']}))"

        if item_config.get('children', None) is not None:
            children = _EmbedString('[' + getContextMenuOpItems(item_config['children']) + ']')

            items.append({
                'name': item_config['name'],
                'tooltip': item_config.get('description', _EmbedString('null')),
                'subMenu': children,
                'disabled': _EmbedString(disabled),
            })

        else:
            if 'builtin' in item_config:
                items.append({
                    'name': item_config['name'],
                    'tooltip': item_config.get('description', _EmbedString('null')),
                    'disabled': _EmbedString(disabled),
                    'action': _EmbedString(getBuiltinAction(item_config))
                })

            else:

                items.append({
                    'name': item_config['name'],
                    'tooltip': item_config.get('description', _EmbedString('null')),
                    'disabled': _EmbedString(disabled),
                    'action': _EmbedString(getOpAction(item_config)),
                })

    return ", \n".join(repr(i) for i in items)
