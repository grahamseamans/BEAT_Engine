# The coupled worker as `coupled_solver.jl` loads it: from the CPU bundle's pkgimage, with the
# precompile workload's native code. It must have run its workload, left no state behind,
# and solve as the driver included from source does.
using Test, JSON, Base64
import BeatEngineCpuBundle

@testset "coupled worker bundle" begin
    worker = BeatEngineCpuBundle.CoupledWorker
    @test startswith(worker.WORKLOAD[], "solved")
    @test isempty(worker.leftover_state())

    source = Core.eval(Main, :(module CoupledBundleParitySource end))
    Base.include(source, normpath(joinpath(@__DIR__, "..", "BeatEngineCoupledDriver.jl")))
    function solve(driver, symmetry)
        request = worker.workload_request(worker.WORKLOAD_FIXTURES, symmetry)
        request["solver_options"]["precision"] = "float64"
        events = mktemp() do path, stream
            redirect_stdout(stream) do
                Base.invokelatest(driver.solve_request, request)
            end
            flush(stream)
            [JSON.parse(line) for line in eachline(path) if startswith(line, "{")]
        end
        return only(event for event in events if haskey(event, "diagnostics"))
    end
    # Every output array, decoded (complex128 at float64). AOT and JIT code differ in the last
    # bits -- solver.jl's note on the exterior bundle: about one ulp -- and nothing more.
    decoded(quantity) = reinterpret(ComplexF64, base64decode(quantity["values"]["content_base64"]))
    for symmetry in ("x", "off")
        from_bundle = solve(worker, symmetry)
        from_source = solve(source, symmetry)
        @test [q["id"] for q in from_bundle["quantities"]] == [q["id"] for q in from_source["quantities"]]
        for (a, b) in zip(from_bundle["quantities"], from_source["quantities"])
            @test a["values"]["dtype"] == "complex128"
            @test isapprox(decoded(a), decoded(b); rtol=1e-10)
        end
    end
end
