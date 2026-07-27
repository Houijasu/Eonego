/// Native graph-evaluator runtime bridge.
///
/// The public contract is deliberately fallback-first: search keeps using NNUE whenever native inference
/// is unavailable or TryEvaluate returns ValueNone.
module Eonego.GraphGpu

open System
open System.Collections.Concurrent
open System.IO
open System.Runtime.InteropServices
open System.Threading
open Eonego.Position

[<Literal>]
let ModelVersion = 1

[<Literal>]
let DefaultBatch = 256

[<Literal>]
let DefaultWaitUs = 200

[<Literal>]
let NativeInputBytes = GraphFeatures.RecordBytes

[<Literal>]
let NativeOutputBytes = 16 + 384 * 4 + 384 * 4

[<Literal>]
let private NativeOk = 0

[<Literal>]
let private NativeNotImplemented = -2

[<Literal>]
let private NativeBackendNone = -1

[<Literal>]
let private NativeBackendCpu = 0

[<Literal>]
let private NativeBackendCuda = 1

let ModelMagic: byte[] = Text.Encoding.ASCII.GetBytes "EONGR01!"

[<Struct>]
type GraphOutput =
    { ValueCp: int
      WdlWin: int
      WdlDraw: int
      WdlLoss: int
      FromLogits: int[]
      ToLogits: int[] }

[<RequireQualifiedAccess>]
type GraphMode =
    | Leaf = 0
    | Policy = 1
    | All = 2

[<RequireQualifiedAccess>]
type NativeBackend =
    | None = -1
    | CpuScalar = 0
    | Cuda = 1

[<AllowNullLiteral; Sealed>]
type GraphNetwork(path: string, schemaHash: uint32, dModel: int, layers: int, flags: int, valueScale: int) =
    member _.Path = path
    member _.SchemaHash = schemaHash
    member _.DModel = dModel
    member _.Layers = layers
    member _.Flags = flags
    member _.ValueScale = valueScale
    member _.HasPolicy = (flags &&& 1) <> 0
    member _.HasWdl = (flags &&& 2) <> 0

type GraphLoadResult =
    | GraphLoaded of GraphNetwork
    | GraphFailed of string

let private readI32 (buf: byte[]) (pos: int) : int =
    int buf.[pos]
    ||| (int buf.[pos + 1] <<< 8)
    ||| (int buf.[pos + 2] <<< 16)
    ||| (int buf.[pos + 3] <<< 24)

let load (path: string) : GraphLoadResult =
    try
        let header = Array.zeroCreate<byte> 36
        use fs = File.OpenRead path

        if fs.Length < int64 header.Length then
            GraphFailed "truncated EONGR01 header"
        else
            let n = fs.Read(header, 0, header.Length)

            if n <> header.Length then
                GraphFailed "could not read EONGR01 header"
            elif header.[0..7] <> ModelMagic then
                GraphFailed "bad magic (want EONGR01)"
            else
                let version = readI32 header 8
                let schema = uint32 (readI32 header 12)
                let dModel = readI32 header 16
                let layers = readI32 header 20
                let flags = readI32 header 24
                let valueScale = readI32 header 28
                let reserved = readI32 header 32

                if version <> ModelVersion then
                    GraphFailed ("unsupported EONGR01 version " + string version)
                elif schema <> GraphFeatures.SchemaHash then
                    GraphFailed "graph feature schema hash mismatch"
                elif dModel < 32 || dModel > 1024 || layers < 1 || layers > 16 then
                    GraphFailed "unsupported graph model dimensions"
                elif valueScale <= 0 || reserved <> 0 then
                    GraphFailed "invalid graph model header"
                else
                    GraphLoaded(GraphNetwork(path, schema, dModel, layers, flags, valueScale))
    with ex ->
        GraphFailed ex.Message

[<Struct; StructLayout(LayoutKind.Sequential)>]
type NativeInputHeader =
    val mutable NodeCount: byte
    val mutable SideToMove: byte
    val mutable InCheck: byte
    val mutable CheckerCount: byte

[<Struct; StructLayout(LayoutKind.Sequential)>]
type NativeOutputHeader =
    val mutable ValueCp: int
    val mutable WdlWin: int
    val mutable WdlDraw: int
    val mutable WdlLoss: int

module private Native =
    [<DllImport("eonego_cuda", CallingConvention = CallingConvention.Cdecl, EntryPoint = "eogpu_create")>]
    extern nativeint Create(string modelPath, int deviceId, int maxBatch, IntPtr errBuf, int errLen)

    [<DllImport("eonego_cuda", CallingConvention = CallingConvention.Cdecl, EntryPoint = "eogpu_infer")>]
    extern int Infer(nativeint handle, IntPtr inputBatch, IntPtr outputBatch, int batchCount)

    [<DllImport("eonego_cuda", CallingConvention = CallingConvention.Cdecl, EntryPoint = "eogpu_backend")>]
    extern int Backend(nativeint handle)

    [<DllImport("eonego_cuda", CallingConvention = CallingConvention.Cdecl, EntryPoint = "eogpu_model_status")>]
    extern int ModelStatus(nativeint handle)

    [<DllImport("eonego_cuda", CallingConvention = CallingConvention.Cdecl, EntryPoint = "eogpu_destroy")>]
    extern void Destroy(nativeint handle)

[<AllowNullLiteral>]
type private PendingEval(pos: Position, needPolicy: bool) =
    let ready = new ManualResetEventSlim(false)
    let mutable result = ValueNone

    member _.Position = pos
    member _.NeedPolicy = needPolicy
    member _.Ready = ready
    member _.Result
        with get () = result
        and set v = result <- v

[<AllowNullLiteral; Sealed>]
type GraphBatcher(network: GraphNetwork, mode: GraphMode, deviceId: int, maxBatch: int, waitUs: int) =
    let maxBatchVal = max 1 maxBatch
    let waitUsVal = max 0 waitUs
    let mutable fallbackCount = 0L
    let mutable nativeFailureCount = 0L
    let mutable nativeBatchCount = 0L
    let mutable queuedBatchCount = 0L
    let mutable queuedRequestCount = 0L
    let mutable maxObservedBatch = 0
    let mutable disabled = 0
    let mutable stopWorker = 0
    let mutable workerStarted = 0
    let mutable workerThread: Thread option = None
    let err = Array.zeroCreate<byte> 512
    let pending = ConcurrentQueue<PendingEval>()
    let wake = new AutoResetEvent(false)

    let createNative () =
        try
            let errHandle = GCHandle.Alloc(err, GCHandleType.Pinned)

            try
                let h = Native.Create(network.Path, deviceId, maxBatchVal, errHandle.AddrOfPinnedObject(), err.Length)

                if h = 0n then
                    0n
                else
                    let rc = Native.Infer(h, IntPtr.Zero, IntPtr.Zero, 0)

                    if rc = NativeOk then
                        h
                    else
                        Native.Destroy h
                        0n
            finally
                errHandle.Free()
        with
        | :? DllNotFoundException
        | :? EntryPointNotFoundException
        | :? BadImageFormatException -> 0n

    let mutable handle = createNative ()

    let disableNative () =
        if Interlocked.Exchange(&disabled, 1) = 0 then
            let h = Interlocked.Exchange(&handle, 0n)
            if h <> 0n then
                Native.Destroy h

    let readI32 (buf: byte[]) (p: int) : int =
        int buf.[p]
        ||| (int buf.[p + 1] <<< 8)
        ||| (int buf.[p + 2] <<< 16)
        ||| (int buf.[p + 3] <<< 24)

    let parseOutput (buf: byte[]) (record: int) (needPolicy: bool) : GraphOutput =
        let basep = record * NativeOutputBytes
        let fromL = if needPolicy then Array.zeroCreate<int> 384 else Array.empty
        let toL = if needPolicy then Array.zeroCreate<int> 384 else Array.empty

        if needPolicy then
            let mutable p = basep + 16

            for i in 0 .. 383 do
                fromL.[i] <- readI32 buf p
                p <- p + 4

            for i in 0 .. 383 do
                toL.[i] <- readI32 buf p
                p <- p + 4

        { ValueCp = readI32 buf basep
          WdlWin = readI32 buf (basep + 4)
          WdlDraw = readI32 buf (basep + 8)
          WdlLoss = readI32 buf (basep + 12)
          FromLogits = fromL
          ToLogits = toL }

    let addFallback n =
        if n > 0 then
            Interlocked.Add(&fallbackCount, int64 n) |> ignore

    let observeBatch count =
        Interlocked.Increment(&nativeBatchCount) |> ignore

        let mutable old = Volatile.Read(&maxObservedBatch)

        while count > old && Interlocked.CompareExchange(&maxObservedBatch, count, old) <> old do
            old <- Volatile.Read(&maxObservedBatch)

    let evaluateDirect(positions: Position[], needPolicy: bool) : GraphOutput[] voption =
        if positions.Length = 0 then
            ValueSome [||]
        else
            let outputs = Array.zeroCreate<GraphOutput> positions.Length
            let mutable start = 0
            let mutable ok = true

            while ok && start < positions.Length do
                let h = Volatile.Read(&handle)

                if h = 0n || Volatile.Read(&disabled) <> 0 then
                    addFallback (positions.Length - start)
                    ok <- false
                else
                    let count = min maxBatchVal (positions.Length - start)
                    let input = Array.zeroCreate<byte> (count * NativeInputBytes)
                    let output = Array.zeroCreate<byte> (count * NativeOutputBytes)

                    // CPU search hands over a dense EONGF01 slab; the native layer sees this as one
                    // bounded batch and later CUDA kernels can replace the current scalar backend.
                    for i in 0 .. count - 1 do
                        let r = GraphFeatures.extract positions.[start + i] |> GraphFeatures.toBytes
                        Buffer.BlockCopy(r, 0, input, i * NativeInputBytes, NativeInputBytes)

                    let inHandle = GCHandle.Alloc(input, GCHandleType.Pinned)
                    let outHandle = GCHandle.Alloc(output, GCHandleType.Pinned)

                    try
                        try
                            observeBatch count
                            let rc = Native.Infer(h, inHandle.AddrOfPinnedObject(), outHandle.AddrOfPinnedObject(), count)

                            if rc = NativeOk then
                                for i in 0 .. count - 1 do
                                    outputs.[start + i] <- parseOutput output i needPolicy

                                start <- start + count
                            else
                                Interlocked.Increment(&nativeFailureCount) |> ignore

                                if rc = NativeNotImplemented then
                                    disableNative ()

                                addFallback (positions.Length - start)
                                ok <- false
                        with
                        | :? DllNotFoundException
                        | :? EntryPointNotFoundException
                        | :? BadImageFormatException ->
                            Interlocked.Increment(&nativeFailureCount) |> ignore
                            disableNative ()
                            addFallback (positions.Length - start)
                            ok <- false
                    finally
                        inHandle.Free()
                        outHandle.Free()

            if ok then ValueSome outputs else ValueNone

    let completeNone (reqs: ResizeArray<PendingEval>) =
        for r in reqs do
            r.Result <- ValueNone
            r.Ready.Set()

    let processRequests (first: PendingEval) =
        let reqs = ResizeArray<PendingEval>(maxBatchVal)
        reqs.Add first

        let deadline =
            if waitUsVal <= 0 then
                0L
            else
                DateTime.UtcNow.Ticks + int64 waitUsVal * 10L

        let mutable draining = true

        while draining && reqs.Count < maxBatchVal do
            let mutable next = Unchecked.defaultof<PendingEval>

            if pending.TryDequeue(&next) then
                reqs.Add next
            elif waitUsVal > 0 then
                let remaining = deadline - DateTime.UtcNow.Ticks

                if remaining > 0L then
                    wake.WaitOne(TimeSpan.FromTicks remaining) |> ignore
                else
                    draining <- false
            else
                draining <- false

        let needPolicy = reqs |> Seq.exists (fun r -> r.NeedPolicy)
        let positions = reqs |> Seq.map (fun r -> r.Position) |> Seq.toArray
        Interlocked.Increment(&queuedBatchCount) |> ignore
        Interlocked.Add(&queuedRequestCount, int64 reqs.Count) |> ignore

        match evaluateDirect(positions, needPolicy) with
        | ValueSome outs when outs.Length = reqs.Count ->
            for i in 0 .. reqs.Count - 1 do
                reqs.[i].Result <- ValueSome outs.[i]
                reqs.[i].Ready.Set()
        | _ -> completeNone reqs

    let drainStopped () =
        let reqs = ResizeArray<PendingEval>()
        let mutable next = Unchecked.defaultof<PendingEval>

        while pending.TryDequeue(&next) do
            reqs.Add next

        completeNone reqs

    let workerLoop () =
        while Volatile.Read(&stopWorker) = 0 do
            let mutable req = Unchecked.defaultof<PendingEval>

            if pending.TryDequeue(&req) then
                processRequests req
            else
                wake.WaitOne() |> ignore

        drainStopped ()

    let ensureWorker () =
        if maxBatchVal > 1 && Interlocked.CompareExchange(&workerStarted, 1, 0) = 0 then
            let t = Thread(ThreadStart(workerLoop))
            t.IsBackground <- true
            t.Name <- "Eonego graph batcher"
            workerThread <- Some t
            t.Start()

    member _.Network = network
    member _.Mode = mode
    member _.DeviceId = deviceId
    member _.MaxBatch = maxBatchVal
    member _.WaitUs = waitUsVal
    member _.FallbackCount = Volatile.Read(&fallbackCount)
    member _.NativeFailureCount = Volatile.Read(&nativeFailureCount)
    member _.NativeBatchCount = Volatile.Read(&nativeBatchCount)
    member _.QueuedBatchCount = Volatile.Read(&queuedBatchCount)
    member _.QueuedRequestCount = Volatile.Read(&queuedRequestCount)
    member _.MaxObservedBatch = Volatile.Read(&maxObservedBatch)
    member _.SupportsInference = Volatile.Read(&disabled) = 0 && Volatile.Read(&handle) <> 0n
    member _.NativeBackend =
        let h = Volatile.Read(&handle)

        if h = 0n || Volatile.Read(&disabled) <> 0 then
            NativeBackend.None
        else
            try
                match Native.Backend h with
                | NativeBackendCuda -> NativeBackend.Cuda
                | NativeBackendCpu -> NativeBackend.CpuScalar
                | _ -> NativeBackend.None
            with
            | :? DllNotFoundException
            | :? EntryPointNotFoundException
            | :? BadImageFormatException -> NativeBackend.None
    member _.HasGraphNetWeights =
        let h = Volatile.Read(&handle)

        if h = 0n || Volatile.Read(&disabled) <> 0 then
            false
        else
            try
                Native.ModelStatus h > 0
            with
            | :? DllNotFoundException
            | :? EntryPointNotFoundException
            | :? BadImageFormatException -> false
    member _.HasLearnedCuda =
        let h = Volatile.Read(&handle)

        if h = 0n || Volatile.Read(&disabled) <> 0 then
            false
        else
            try
                Native.ModelStatus h = 2
            with
            | :? DllNotFoundException
            | :? EntryPointNotFoundException
            | :? BadImageFormatException -> false

    member this.TryEvaluate(pos: Position, needPolicy: bool) : GraphOutput voption =
        if maxBatchVal > 1 && this.SupportsInference && Volatile.Read(&stopWorker) = 0 then
            ensureWorker ()
            let req = PendingEval(pos, needPolicy)
            pending.Enqueue req
            wake.Set() |> ignore
            req.Ready.Wait()
            let r = req.Result
            req.Ready.Dispose()
            r
        else
            match this.TryEvaluateBatch([| pos |], needPolicy) with
            | ValueSome batch when batch.Length = 1 -> ValueSome batch.[0]
            | _ -> ValueNone

    member this.TryEvaluateBatch(positions: Position[], needPolicy: bool) : GraphOutput[] voption =
        ignore this
        evaluateDirect(positions, needPolicy)

    interface IDisposable with
        member _.Dispose() =
            Interlocked.Exchange(&stopWorker, 1) |> ignore
            wake.Set() |> ignore

            match workerThread with
            | Some t when t.IsAlive -> t.Join 1000 |> ignore
            | _ -> ()

            disableNative ()
            wake.Dispose()
