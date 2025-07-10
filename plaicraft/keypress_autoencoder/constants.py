# constants.py

# Store the id_to_name and id_to_index mappings here

# Keyboard keys with string IDs
FIXED_KEYS = {
    "32": "space", "65": "a", "66": "b", "67": "c", "68": "d", "69": "e", "70": "f",
    "71": "g", "72": "h", "73": "i", "74": "j", "75": "k", "76": "l", "77": "m",
    "78": "n", "79": "o", "80": "p", "81": "q", "82": "r", "83": "s", "84": "t",
    "85": "u", "86": "v", "87": "w", "88": "x", "89": "y", "90": "z", "48": "0",
    "49": "1", "50": "2", "51": "3", "52": "4", "53": "5", "54": "6", "55": "7",
    "56": "8", "57": "9", "256": "Escape", "257": "Return", "258": "Tab", "259": "BackSpace",
    "260": "Insert", "261": "Delete", "262": "Right", "263": "Left", "264": "Down",
    "265": "Up", "266": "Page_Up", "267": "Page_Down", "268": "Home", "269": "End",
    "340": "Shift_L", "341": "Control_L", "342": "Alt_L", "343": "Super_L", "344": "Shift_R",
    "345": "Control_R", "346": "Alt_R", "347": "Super_R", "348": "Menu", "91": "bracketleft",
    "93": "bracketright", "92": "backslash", "59": "semicolon", "39": "apostrophe",
    "44": "comma", "46": "period", "47": "slash", "45": "minus", "61": "equal", "96": "grave",
    "290": "F1", "292": "F3", "294": "F5", "280": "Caps_Lock"
}

# Mouse buttons and scroll events with string identifiers
FIXED_MOUSE_BUTTONS = {
    "left": "mouse_left",
    "right": "mouse_right",
    "scroll_up": "scroll_up",
    "scroll_down": "scroll_down"
}

id_to_name = {}
id_to_index = {}
idx = 0

# Populate keyboard keys
for key_id, key_name in FIXED_KEYS.items():
    id_to_name[key_id] = key_name
    id_to_index[key_id] = idx
    idx += 1

# Populate mouse buttons
for btn_id, btn_name in FIXED_MOUSE_BUTTONS.items():
    id_to_name[btn_id] = btn_name
    id_to_index[btn_id] = idx
    idx += 1
