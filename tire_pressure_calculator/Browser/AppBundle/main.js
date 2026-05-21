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

const dotnetRuntime = await dotnet
    .withDiagnosticTracing(false)
    .withApplicationArgumentsFromQuery()
    .create();

const config = dotnetRuntime.getConfig();

await dotnetRuntime.runMain(config.mainAssemblyName, [window.location.search]);
