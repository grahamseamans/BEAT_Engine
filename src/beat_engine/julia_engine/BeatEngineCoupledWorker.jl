#=
The coupled worker's half of a bundle. Each `BeatEngine*Bundle` includes this
file at its top level, after it has loaded `BeatEngineCore` and named its
backend, so the module below is `BeatEngine*Bundle.CoupledWorker`, the module
`julia_local/coupled_solver.jl` runs.

It holds `julia_local/BeatEngineCoupledDriver.jl` -- the coupled driver, which
imports the bundle's engine rather than compiling a second copy -- and a
precompile workload: one float32 frequency of the frozen coupled reference
fixture, with X symmetry and without (`julia_local/tests/fixtures`, an 842-vertex FEM volume behind a
conforming BEM surface) on the CPU backend. The CPU backend in every bundle,
for the reason the exterior workload gives: precompilation runs where there
may be no accelerator, and the CPU path compiles everything up to the backend
branch -- mesh loading, request parsing, the FEM system, the output builders.
Device kernels are compiled by the running process in any case.

`WORKLOAD[]` is serialized with the pkgimage and says what the workload did:
"solved ...", "skipped: ..." or "failed: ...". A skip or a failure is also
logged once, when the cache is built. A failure does not fail the build -- the
bundle still loads and solves, compiling at run time what the workload did not
reach -- so the image build checks `WORKLOAD[]` itself.

A solve leaves state behind in module globals (provenance, the mesh record,
field caches, MUMPS handles) and the pkgimage would serialize it, so the
workload resets it afterwards and refuses to finish if any of it survives.
=#
module CoupledWorker

using PrecompileTools: @compile_workload

const BEAT_ENGINE_BACKEND = parentmodule(@__MODULE__).BEAT_ENGINE_BACKEND
import ..BeatEngineCore

include(joinpath(parentmodule(@__MODULE__).ENGINE_DIR, "BeatEngineCoupledDriver.jl"))

# Read at load time by `BeatEngineContract` and so baked into the image: editing
# them has to rebuild it, as editing an included source does.
include_dependency(joinpath(BeatEngineContract.BeatEngineProvenance.ENGINE_ROOT, "..", "beat_contract", "system-v1.schema.json"))
include_dependency(joinpath(BeatEngineContract.BeatEngineProvenance.ENGINE_ROOT, "..", "beat_contract", "worker-v1.json"))

const WORKLOAD = Ref{String}("not run")

const WORKLOAD_FIXTURES = joinpath(parentmodule(@__MODULE__).ENGINE_DIR, "tests", "fixtures")

"""
One frequency of the reference fixture, shaped like what Boundary Lab submits for a cabinet
(2026-10-05: polar `exterior_pressure` split into observation domains, interface velocity and
interface-radiated pressure; float32, static condensation, frequency-invariant caches,
order-2 quadrature, the e^{+iωt} convention). `symmetry = "x"` moves both meshes 0.5 m into
the positive-x half space, where the X-symmetric solve requires them.
"""
function workload_request(fixtures, symmetry)
    offset = symmetry == "x" ? [0.5, 0.0, 0.0] : [0.0, 0.0, 0.0]
    group(mesh_id, tag) = Dict{String,Any}("mesh_id" => mesh_id, "dimension" => 2, "tag" => tag, "name" => nothing)
    boundary(id, kind, region, mesh_id, tag) = Dict{String,Any}("id" => id, "name" => id, "kind" => kind,
        "region_id" => region, "group" => group(mesh_id, tag), "parameters" => Dict{String,Any}())
    system = Dict{String,Any}("id" => "system:precompile-workload", "name" => "precompile workload", "contract_version" => 1,
        "meshes" => Any[
            Dict{String,Any}("id" => "mesh:exterior", "name" => "Exterior", "file" => joinpath(fixtures, "exterior_conforming.msh"),
                "purpose" => "bem_surface", "scale_to_m" => 0.001, "translation_m" => offset),
            Dict{String,Any}("id" => "mesh:interior", "name" => "Interior", "file" => joinpath(fixtures, "femvolume.msh"),
                "purpose" => "fem_volume", "scale_to_m" => 0.001, "translation_m" => offset),
        ],
        "regions" => Any[
            Dict{String,Any}("id" => "region:exterior", "name" => "Exterior", "kind" => "unbounded_air", "mesh_ids" => ["mesh:exterior"],
                "volume_groups" => Any[], "sound_speed_m_per_s" => 343.0, "density_kg_per_m3" => 1.21, "loss_model" => Dict{String,Any}()),
            Dict{String,Any}("id" => "region:interior", "name" => "Interior", "kind" => "bounded_air", "mesh_ids" => ["mesh:interior"],
                "volume_groups" => Any[Dict{String,Any}("mesh_id" => "mesh:interior", "dimension" => 3, "tag" => 1, "name" => nothing)],
                "sound_speed_m_per_s" => 343.0, "density_kg_per_m3" => 1.21, "loss_model" => Dict{String,Any}()),
        ],
        "boundaries" => Any[
            boundary("boundary:radiator", "moving", "region:interior", "mesh:interior", 2),
            boundary("boundary:interior-interface", "interface", "region:interior", "mesh:interior", 3),
            boundary("boundary:walls", "rigid", "region:interior", "mesh:interior", 4),
            boundary("boundary:exterior-box", "rigid", "region:exterior", "mesh:exterior", 1),
            boundary("boundary:exterior-interface", "interface", "region:exterior", "mesh:exterior", 2),
        ],
        "interfaces" => Any[Dict{String,Any}("id" => "interface:one", "name" => "One",
            "bounded_boundary_id" => "boundary:interior-interface", "unbounded_boundary_id" => "boundary:exterior-interface",
            "topology" => Dict{String,Any}("fem_vertex_indices" => [0], "fem_to_bem_vertex_indices" => [0], "fem_face_indices" => [0],
                "bem_face_indices" => [0], "normal_sign" => [-1], "max_coordinate_error" => 0.0,
                "fem_facets_on_tetra_boundary" => 1, "bem_boundary_edges" => 0))],
        "components" => Any[Dict{String,Any}("id" => "component:radiator", "name" => "Radiator", "kind" => "ideal_velocity_source",
            "boundary_ids" => ["boundary:radiator"], "parameters" => Dict{String,Any}())],
        "excitation_ports" => Any[Dict{String,Any}("id" => "excitation:radiator", "name" => "Radiator",
            "component_id" => "component:radiator", "kind" => "normal_velocity")],
    )
    polar = [[2.0 * sin(angle), 0.0, 2.0 * cos(angle)] for angle in range(0, pi; length=7)]
    outputs = Any[
        Dict{String,Any}("id" => "pressure", "quantity" => "exterior_pressure", "target_ids" => Any[],
            "options" => Dict{String,Any}("points_m" => polar, "observation_domains" => Any[
                Dict{String,Any}("id" => "observation:a", "quantity_id" => "pressure:a", "offset" => 0, "count" => 4),
                Dict{String,Any}("id" => "observation:b", "quantity_id" => "pressure:b", "offset" => 4, "count" => 3)])),
        Dict{String,Any}("id" => "velocity", "quantity" => "interface_average_normal_velocity", "target_ids" => Any[],
            "options" => Dict{String,Any}()),
        Dict{String,Any}("id" => "radiated", "quantity" => "interface_radiated_pressure", "target_ids" => Any[],
            "options" => Dict{String,Any}("points_m" => Any[[0.0, 0.0, 2.0]])),
    ]
    return Dict{String,Any}("schema_version" => 1, "compiled_system" => system, "frequencies_hz" => [500.0],
        "excitation_port_ids" => ["excitation:radiator"], "outputs" => outputs,
        "solver_options" => Dict{String,Any}("precision" => "float32", "bem_backend" => "cpu",
            "quadrature_order" => 2, "singular_order" => 2, "validation_diagnostics" => false,
            "cache_frequency_invariant" => true, "static_condensation" => true, "symmetry" => symmetry,
            "phasor_convention" => "exp(+i omega t)"))
end

"""Module state a solve may leave behind that must not reach the pkgimage; empty when clean."""
function leftover_state()
    leftovers = String[]
    provenance = BeatEngineContract.BeatEngineProvenance
    provenance.IDENTITY[] === nothing || push!(leftovers, "BeatEngineProvenance.IDENTITY")
    provenance.RUNTIME[] === nothing || push!(leftovers, "BeatEngineProvenance.RUNTIME")
    isempty(RUN_MESH_PROVENANCE[]) || push!(leftovers, "RUN_MESH_PROVENANCE")
    isempty(BEM_FIELD_EVALUATION_CACHES) || push!(leftovers, "BEM_FIELD_EVALUATION_CACHES")
    isempty(BEM_FIELD_EVALUATION_CACHE_ORDER) || push!(leftovers, "BEM_FIELD_EVALUATION_CACHE_ORDER")
    mumps = BeatEngineCoupledCondensed.BeatEngineMumps
    mumps.LIBRARY[] === nothing || push!(leftovers, "BeatEngineMumps.LIBRARY")
    mumps.FORCE_UNAVAILABLE[] && push!(leftovers, "BeatEngineMumps.FORCE_UNAVAILABLE")
    isempty(mumps.LIVE_SOLVERS) || push!(leftovers, "BeatEngineMumps.LIVE_SOLVERS")
    mumps.ATEXIT_REGISTERED[] && push!(leftovers, "BeatEngineMumps.ATEXIT_REGISTERED")
    return leftovers
end

function reset_after_workload!()
    provenance = BeatEngineContract.BeatEngineProvenance
    provenance.IDENTITY[] = nothing
    provenance.RUNTIME[] = nothing
    RUN_MESH_PROVENANCE[] = []
    release_all_bem_field_evaluation_caches!()
    GC.gc(true)
    leftovers = leftover_state()
    isempty(leftovers) || error("BEAT coupled precompile workload left state that would be baked into the pkgimage: " *
        join(leftovers, ", ") * ". Reset it in reset_after_workload!.")
    return nothing
end

let leftovers = leftover_state()
    # The driver itself must load clean, or the reset above is not enough.
    isempty(leftovers) || error("BEAT coupled driver loaded with non-empty state: " * join(leftovers, ", "))
end

if !isdir(WORKLOAD_FIXTURES)
    WORKLOAD[] = "skipped: no fixture directory at $(WORKLOAD_FIXTURES)"
    @warn "BEAT coupled precompile workload $(WORKLOAD[]); the first coupled frequency will compile at run time."
else
    @compile_workload begin
        started = time()
        try
            for symmetry in ("x", "off")
                redirect_stdout(devnull) do
                    solve_request(workload_request(WORKLOAD_FIXTURES, symmetry))
                end
            end
            WORKLOAD[] = "solved: one float32 frequency of the reference fixture, symmetry x and off, " *
                "$(round(time() - started; digits=1)) s"
        catch exception
            WORKLOAD[] = "failed: " * sprint(showerror, exception)
            @error "BEAT coupled precompile workload failed; the bundle still loads, the first coupled frequency compiles at run time." exception = (exception, catch_backtrace())
        finally
            reset_after_workload!()
        end
        # The worker's own loop reads stdin and never runs here; asking for it by signature
        # compiles its dispatch into `solve_request`, as the exterior workload does for its loop.
        precompile(run_worker, ())
    end
end

end
