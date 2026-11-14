using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Avalonia.Threading;

namespace InfernoAnalyzer.App;

/// <summary>
/// Reports a frozen window. A hang leaves nothing behind: no exception, no
/// log, and a NativeAOT executable has no managed debugger to attach. So a
/// background thread posts a heartbeat to the UI thread every second, and
/// when one has gone unanswered for <see cref="Threshold"/> it suspends the UI
/// thread for a moment, walks its native call stack and appends it to the
/// crash log as module names and offsets (the build's debug symbols turn
/// those into function names). Once per hang; a recovery is logged too.
/// Windows on x64 only: the stack walk uses the Windows unwinder.
/// </summary>
internal static unsafe partial class HangWatch
{
    /// <summary>How long the UI thread may go without answering before it is reported.</summary>
    internal static readonly TimeSpan Threshold = TimeSpan.FromSeconds(5);

    private const int MaxFrames = 128;
    private const int ContextSize = 1232; // sizeof(CONTEXT) on x64; it must be 16-byte aligned
    private const int ContextFlagsOffset = 0x30;
    private const int RspOffset = 0x98;
    private const int RipOffset = 0xF8;
    private const uint ContextControlAndInteger = 0x00100003; // CONTEXT_AMD64 | CONTEXT_CONTROL | CONTEXT_INTEGER
    private const uint ThreadAccess = 0x0002 | 0x0008 | 0x0040; // SUSPEND_RESUME | GET_CONTEXT | QUERY_INFORMATION

    private static long _lastBeat;
    private static int _beatPending;

    /// <summary>True where a stack can be captured.</summary>
    internal static bool IsSupported => OperatingSystem.IsWindows() && RuntimeInformation.ProcessArchitecture == Architecture.X64;

    /// <summary>The calling thread's operating-system id.</summary>
    internal static uint CurrentThreadId() => GetCurrentThreadId();

    /// <summary>Start watching the calling thread, which must be the UI thread. Call once the app has started.</summary>
    public static void Start()
    {
        if (!IsSupported)
        {
            return;
        }

        var uiThread = GetCurrentThreadId();
        Volatile.Write(ref _lastBeat, Environment.TickCount64);
        new Thread(() => Watch(uiThread)) { IsBackground = true, Name = "Hang watch" }.Start();
    }

    private static void Watch(uint uiThread)
    {
        var reported = false;
        while (true)
        {
            Thread.Sleep(1000);
            if (Interlocked.Exchange(ref _beatPending, 1) == 0)
            {
                // Ahead of everything else queued: only a UI thread that runs nothing at all misses it.
                Dispatcher.UIThread.Post(
                    () =>
                    {
                        Volatile.Write(ref _lastBeat, Environment.TickCount64);
                        Volatile.Write(ref _beatPending, 0);
                    },
                    DispatcherPriority.Send);
            }

            var silent = TimeSpan.FromMilliseconds(Environment.TickCount64 - Volatile.Read(ref _lastBeat));
            if (silent < Threshold)
            {
                if (reported)
                {
                    CrashLog.WriteText("The UI thread is responding again", string.Empty);
                }

                reported = false;
                continue;
            }

            if (!reported)
            {
                reported = true;
                CrashLog.WriteText(
                    $"The UI thread has not responded for {silent.TotalSeconds.ToString("0", CultureInfo.InvariantCulture)} s; its call stack",
                    CaptureStack(uiThread));
            }
        }
    }

    /// <summary>
    /// The native call stack of thread <paramref name="threadId"/> (not the calling thread), innermost
    /// frame first, one "module+0xoffset" line per frame. Every frame but the first is a return address.
    /// </summary>
    internal static string CaptureStack(uint threadId)
    {
        if (!IsSupported)
        {
            return "(not supported on this platform)";
        }

        var addresses = stackalloc ulong[MaxFrames];
        var count = 0;
        var context = (byte*)NativeMemory.AlignedAlloc(ContextSize, 16);
        var thread = OpenThread(ThreadAccess, 0, threadId);
        if (thread == 0)
        {
            NativeMemory.AlignedFree(context);
            return $"(could not open thread {threadId}: error {Marshal.GetLastPInvokeError()})";
        }

        // While the other thread is suspended this one must not wait for a garbage collection: the
        // collector would in turn wait for the suspended thread. Nothing below allocates, and the
        // no-GC region keeps a collection started by another thread out of these few microseconds.
        var noGc = false;
        try
        {
            noGc = GC.TryStartNoGCRegion(16 * 1024 * 1024);
        }
        catch (InvalidOperationException)
        {
            // Already in a no-GC region.
        }

        try
        {
            if (SuspendThread(thread) != uint.MaxValue)
            {
                try
                {
                    NativeMemory.Clear(context, ContextSize);
                    *(uint*)(context + ContextFlagsOffset) = ContextControlAndInteger;
                    if (GetThreadContext(thread, context) != 0)
                    {
                        count = Unwind(context, addresses);
                    }
                }
                finally
                {
                    ResumeThread(thread);
                }
            }
        }
        finally
        {
            if (noGc)
            {
                try
                {
                    GC.EndNoGCRegion();
                }
                catch (InvalidOperationException)
                {
                    // The region already ended (the allocation budget was used up elsewhere).
                }
            }

            CloseHandle(thread);
            NativeMemory.AlignedFree(context);
        }

        return count == 0 ? "(no frames)" : Describe(addresses, count);
    }

    /// <summary>Unwind from <paramref name="context"/> with the Windows unwinder; no allocation, see <see cref="CaptureStack"/>.</summary>
    private static int Unwind(byte* context, ulong* addresses)
    {
        var count = 0;
        var process = GetCurrentProcess();
        while (count < MaxFrames)
        {
            var rip = *(ulong*)(context + RipOffset);
            if (rip == 0)
            {
                break;
            }

            addresses[count++] = rip;
            ulong imageBase;
            var function = RtlLookupFunctionEntry(rip, &imageBase, null);
            if (function is null)
            {
                // A leaf function has no unwind data: its return address is at the top of the stack.
                var rsp = *(ulong*)(context + RspOffset);
                ulong returnAddress;
                nuint read;
                if (ReadProcessMemory(process, (void*)rsp, &returnAddress, sizeof(ulong), &read) == 0 || read != sizeof(ulong))
                {
                    break;
                }

                *(ulong*)(context + RipOffset) = returnAddress;
                *(ulong*)(context + RspOffset) = rsp + sizeof(ulong);
            }
            else
            {
                void* handlerData;
                ulong establisherFrame;
                RtlVirtualUnwind(0, imageBase, rip, function, context, &handlerData, &establisherFrame, null);
            }
        }

        return count;
    }

    private static string Describe(ulong* addresses, int count)
    {
        var text = new StringBuilder();
        var name = stackalloc char[520];
        for (var i = 0; i < count; i++)
        {
            nint module;
            // FROM_ADDRESS | UNCHANGED_REFCOUNT
            if (GetModuleHandleExW(0x4 | 0x2, (void*)addresses[i], &module) != 0 && module != 0)
            {
                var length = GetModuleFileNameW(module, name, 520);
                var file = Path.GetFileName(new string(name, 0, (int)length));
                text.Append(CultureInfo.InvariantCulture, $"  #{i:00} {file}+0x{addresses[i] - (ulong)module:x}").AppendLine();
            }
            else
            {
                text.Append(CultureInfo.InvariantCulture, $"  #{i:00} 0x{addresses[i]:x}").AppendLine();
            }
        }

        return text.ToString();
    }

    [LibraryImport("kernel32")]
    private static partial uint GetCurrentThreadId();

    [LibraryImport("kernel32")]
    private static partial nint GetCurrentProcess();

    [LibraryImport("kernel32", SetLastError = true)]
    private static partial nint OpenThread(uint access, int inheritHandle, uint threadId);

    [LibraryImport("kernel32")]
    private static partial int CloseHandle(nint handle);

    [LibraryImport("kernel32")]
    private static partial uint SuspendThread(nint thread);

    [LibraryImport("kernel32")]
    private static partial uint ResumeThread(nint thread);

    [LibraryImport("kernel32")]
    private static partial int GetThreadContext(nint thread, byte* context);

    [LibraryImport("kernel32")]
    private static partial int ReadProcessMemory(nint process, void* address, void* buffer, nuint size, nuint* read);

    [LibraryImport("kernel32")]
    private static partial void* RtlLookupFunctionEntry(ulong controlPc, ulong* imageBase, void* historyTable);

    [LibraryImport("kernel32")]
    private static partial void* RtlVirtualUnwind(uint handlerType, ulong imageBase, ulong controlPc, void* functionEntry, byte* context, void** handlerData, ulong* establisherFrame, void* contextPointers);

    [LibraryImport("kernel32")]
    private static partial int GetModuleHandleExW(uint flags, void* address, nint* module);

    [LibraryImport("kernel32")]
    private static partial uint GetModuleFileNameW(nint module, char* fileName, uint size);
}
