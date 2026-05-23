// Run test classes sequentially within this assembly. The C# tire-pressure
// calculator's ViewModels touch two process-wide singletons that don't
// tolerate concurrent mutation:
//
//   - Localizer.Instance — language is global, and language flips fire
//     PropertyChanged(string.Empty) on every subscribed TireCornerViewModel
//     ever instantiated in the process. Tests in another class flipping
//     language race with the assertions of tests creating corner VMs.
//   - AppSettings.Storage (file-backed) — every MainViewModel ctor reads
//     and every setter writes the same on-disk settings.json, so two
//     parallel tests fight over its contents.
//
// Disabling parallel test-class execution removes both races and keeps
// the suite deterministic. Cost is tiny: the whole suite is ~80 fast tests.
[assembly: Xunit.CollectionBehavior(DisableTestParallelization = true)]
