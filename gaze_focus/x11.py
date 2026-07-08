"""X11 helpers: monitor layout, window enumeration, activation, keyboard state."""
import re
import subprocess
from dataclasses import dataclass

from Xlib import X, XK, display, protocol
from Xlib.error import XError
from Xlib.ext import xtest

SKIP_WINDOW_TYPES = ("DOCK", "DESKTOP", "NOTIFICATION", "TOOLTIP", "MENU",
                     "DROPDOWN_MENU", "POPUP_MENU", "SPLASH", "COMBO")
ALL_DESKTOPS = 0xFFFFFFFF


@dataclass
class Monitor:
    name: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class Win:
    id: int
    title: str
    x: int
    y: int
    width: int
    height: int

    def contains(self, px, py):
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


def get_monitors():
    out = subprocess.check_output(["xrandr", "--listmonitors"], text=True)
    mons = []
    for m in re.finditer(
            r"^\s*\d+:\s+\S+\s+(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)\s+(\S+)\s*$",
            out, re.M):
        w, h, x, y = (int(m.group(i)) for i in range(1, 5))
        mons.append(Monitor(m.group(5), x, y, w, h))
    if not mons:
        raise RuntimeError("no monitors found via xrandr --listmonitors")
    return mons


class X11:
    def __init__(self):
        self.d = display.Display()
        self.root = self.d.screen().root
        self._atoms = {}

    def atom(self, name):
        if name not in self._atoms:
            self._atoms[name] = self.d.intern_atom(name)
        return self._atoms[name]

    def _prop(self, win, name):
        try:
            return win.get_full_property(self.atom(name), X.AnyPropertyType)
        except XError:
            return None

    def _prop_values(self, win, name):
        p = self._prop(win, name)
        return list(p.value) if p and p.value is not None else []

    def _title(self, win):
        for name in ("_NET_WM_NAME", "WM_NAME"):
            p = self._prop(win, name)
            if p and p.value:
                v = p.value
                return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
        return "?"

    def _abs_position(self, win):
        """Absolute root-space position via parent walk (frame offsets included)."""
        x = y = 0
        node = win
        while node.id != self.root.id:
            geo = node.get_geometry()
            x += geo.x
            y += geo.y
            node = node.query_tree().parent
        return x, y

    def current_desktop(self):
        vals = self._prop_values(self.root, "_NET_CURRENT_DESKTOP")
        return vals[0] if vals else None

    def _skip(self, win, current_desktop):
        types = self._prop_values(win, "_NET_WM_WINDOW_TYPE")
        for t in SKIP_WINDOW_TYPES:
            if self.atom(f"_NET_WM_WINDOW_TYPE_{t}") in types:
                return True
        if self.atom("_NET_WM_STATE_HIDDEN") in self._prop_values(win, "_NET_WM_STATE"):
            return True
        desk = self._prop_values(win, "_NET_WM_DESKTOP")
        if (desk and current_desktop is not None
                and desk[0] not in (current_desktop, ALL_DESKTOPS)):
            return True
        return False

    def list_windows(self):
        """Visible normal windows on the current workspace, topmost first."""
        ids = self._prop_values(self.root, "_NET_CLIENT_LIST_STACKING")
        cur = self.current_desktop()
        wins = []
        for wid in reversed(ids):  # stacking list is bottom-to-top
            win = self.d.create_resource_object("window", wid)
            try:
                if self._skip(win, cur):
                    continue
                x, y = self._abs_position(win)
                geo = win.get_geometry()
                if geo.width < 40 or geo.height < 40:
                    continue
                wins.append(Win(wid, self._title(win), x, y, geo.width, geo.height))
            except XError:
                continue  # window vanished mid-enumeration
        return wins

    def active_window(self):
        vals = self._prop_values(self.root, "_NET_ACTIVE_WINDOW")
        return vals[0] if vals else None

    def activate(self, wid):
        win = self.d.create_resource_object("window", wid)
        # source indication 2 (pager) sidesteps focus-stealing prevention
        ev = protocol.event.ClientMessage(
            window=win, client_type=self.atom("_NET_ACTIVE_WINDOW"),
            data=(32, [2, X.CurrentTime, 0, 0, 0]))
        self.root.send_event(
            ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        self.d.flush()

    def any_key_down(self):
        return any(self.d.query_keymap())

    def click(self, x, y, button=1):
        """Warp the pointer to (x, y) and click there via XTest."""
        self.root.warp_pointer(int(x), int(y))
        self.d.sync()
        xtest.fake_input(self.d, X.ButtonPress, button)
        xtest.fake_input(self.d, X.ButtonRelease, button)
        self.d.flush()

    def grab_key(self, key_name):
        """Grab a key globally (all modifier-lock variants). Returns keycode."""
        keysym = XK.string_to_keysym(key_name)
        if keysym == 0:
            raise SystemExit(f"unknown key name: {key_name}")
        keycode = self.d.keysym_to_keycode(keysym)
        for mods in (0, X.Mod2Mask, X.LockMask, X.Mod2Mask | X.LockMask):
            self.root.grab_key(keycode, mods, True, X.GrabModeAsync, X.GrabModeAsync)
        self.d.flush()
        return keycode

    def grabbed_key_presses(self):
        """Keycodes of grabbed keys pressed since the last call (non-blocking)."""
        codes = []
        while self.d.pending_events():
            ev = self.d.next_event()
            if ev.type == X.KeyPress:
                codes.append(ev.detail)
        return codes
