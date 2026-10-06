#!/usr/bin/env julia
"""
The BEAT coupled worker entry point: loading and dispatch only.

The driver is in `BeatEngineCoupledDriver.jl`. It is loaded from the bundle
package for this process's accelerator (`julia_engine/BeatEngine*Bundle`,
module `CoupledWorker`), whose pkgimage holds its native code and that of a
precompile workload's coupled solve, or, when no bundle loads, included from
source and compiled in this process, as before the bundles existed.

`BLAB_BEAT_ENGINE_BUNDLE=0` forces the source path, as it does for `solver.jl`.
Either fallback says so on stderr once, with the reason: a worker silently back
on the slow path would look like a regression with no cause.
"""

using JSON

#: Same choice as `solver.jl`: the variable the Python wrapper sets wins, the
#: project directory is the fallback.
const BEAT_ENGINE_BUNDLE_NAME = let
    hint = lowercase(strip(get(ENV, "BLAB_BEAT_ENGINE_GPU_BACKEND", "")))
    if isempty(hint)
        active = Base.active_project()
        directory = active === nothing ? "" : lowercase(basename(dirname(active)))
        hint = directory == "julia_cuda" ? "cuda" :
            directory == "julia_rocm" ? "rocm" :
            directory == "julia_metal" ? "metal" : "cpu"
    end
    hint == "cuda" ? :BeatEngineCudaBundle :
        hint == "rocm" ? :BeatEngineRocmBundle :
        hint == "metal" ? :BeatEngineMetalBundle : :BeatEngineCpuBundle
end

const BEAT_COUPLED_DRIVER = let
    reason = if get(ENV, "BLAB_BEAT_ENGINE_BUNDLE", "1") == "0"
        "BLAB_BEAT_ENGINE_BUNDLE=0"
    else
        try
            @eval using $BEAT_ENGINE_BUNDLE_NAME
            @eval $BEAT_ENGINE_BUNDLE_NAME.CoupledWorker
        catch exception
            "$(BEAT_ENGINE_BUNDLE_NAME) did not load: $(sprint(showerror, exception))"
        end
    end
    if reason isa Module
        reason
    else
        println(stderr, "BEAT coupled worker: compiling the engine from source (", reason, ").")
        flush(stderr)
        driver = Core.eval(Main, :(module BeatEngineCoupledFromSource end))
        Base.include(driver, joinpath(@__DIR__, "BeatEngineCoupledDriver.jl"))
        driver
    end
end

if "--worker" in ARGS
    try
        Base.invokelatest(BEAT_COUPLED_DRIVER.run_worker)
    catch exception
        Base.invokelatest(BEAT_COUPLED_DRIVER.reclaim_accelerator_memory!)
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        exit(1)
    end
else
    try
        request = JSON.parse(read(stdin, String))
        Base.invokelatest(BEAT_COUPLED_DRIVER.solve_request, request)
    catch exception
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        exit(1)
    end
end
