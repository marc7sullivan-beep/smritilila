# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
# Dot-sourced by the restart scripts (restart_worlds.ps1, restart_room.ps1, restart_waker.ps1): report any console or terminal window that appears while they run.
#   . (Join-Path $PSScriptRoot "silence_check.ps1")
#   Start-SilenceWatch      # remember the windows already on screen
#   Watch-Windows           # call it often (every 150 ms) while the restart runs: a flash is caught too
#   Report-Windows          # the line to print at the end
# It looks at window CLASSES, not process names (2026-10-15): the window a console program opens belongs to WindowsTerminal.exe here (the default terminal), and the check
# that looked at python/powershell/cmd processes said "none" over six windows titled python.exe cascaded on screen. A new console of any kind is a window of one of these classes.
if (-not ('SmritililaWindows' -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
public static class SmritililaWindows {
    delegate bool EnumProc(IntPtr h, IntPtr l);
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc p, IntPtr l);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetClassName(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
    // handle|class|title of every visible console or Windows Terminal window
    public static List<string> Consoles() {
        var r = new List<string>();
        EnumWindows((h, l) => {
            var c = new StringBuilder(256); GetClassName(h, c, 256);
            string k = c.ToString();
            if ((k == "CASCADIA_HOSTING_WINDOW_CLASS" || k == "ConsoleWindowClass") && IsWindowVisible(h)) {
                var t = new StringBuilder(256); GetWindowText(h, t, 256);
                r.Add(h.ToInt64() + "|" + k + "|" + t);
            }
            return true;
        }, IntPtr.Zero);
        return r;
    }
}
"@
}

function Start-SilenceWatch {
    $script:silBase = @{ }
    $script:silSeen = @{ }
    foreach ($w in [SmritililaWindows]::Consoles()) { $script:silBase[$w.Split('|')[0]] = $true }
}

function Watch-Windows {
    foreach ($w in [SmritililaWindows]::Consoles()) {
        $id = $w.Split('|')[0]
        if (-not $script:silBase.ContainsKey($id) -and -not $script:silSeen.ContainsKey($id)) { $script:silSeen[$id] = $w.Substring($id.Length + 1) }
    }
}

function Report-Windows {
    Watch-Windows
    "console or terminal windows that appeared during the restart: " + $(if ($script:silSeen.Count) { ($script:silSeen.Values -join "; ") } else { "none" })
}
