import { dotnet } from './_framework/dotnet.js'

const is_browser = typeof window != "undefined";
if (!is_browser) throw new Error("Expected to be running in a browser");

// Hide the loading spinner as soon as Avalonia inserts its canvas into #out.
const out = document.getElementById("out");
const loading = document.getElementById("loading");
if (out && loading) {
    const observer = new MutationObserver(() => {
        if (out.querySelector("canvas")) {
            loading.remove();
            observer.disconnect();
        }
    });
    observer.observe(out, { childList: true, subtree: true });
}

// Avalonia uses a hidden HTML input for IME / virtual-keyboard handling.
// Every value in this app is numeric, so hint a decimal keypad so phones
// don't open a full QWERTY layout.
const hintDecimalKeypad = (root) => {
    if (!(root instanceof Element)) return;
    if (root.matches?.("input, textarea") && !root.inputMode) {
        root.inputMode = "decimal";
    }
    for (const el of root.querySelectorAll?.("input, textarea") ?? []) {
        if (!el.inputMode) el.inputMode = "decimal";
    }
};
const inputObserver = new MutationObserver((mutations) => {
    for (const m of mutations) {
        for (const node of m.addedNodes) hintDecimalKeypad(node);
    }
});
inputObserver.observe(document.body, { childList: true, subtree: true });

const dotnetRuntime = await dotnet
    .withDiagnosticTracing(false)
    .withApplicationArgumentsFromQuery()
    .create();

const config = dotnetRuntime.getConfig();

await dotnetRuntime.runMain(config.mainAssemblyName, [window.location.search]);
