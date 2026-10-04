using Test, JSON

include(joinpath(@__DIR__, "..", "src", "BeatEngineContract.jl"))
using .BeatEngineContract

const CONTRACT_CORPUS = JSON.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "conformance.json"))

@testset "BEAT compiled-system wire conformance" begin
    for case in CONTRACT_CORPUS["cases"]
        @testset "$(case["name"])" begin
            request = deepcopy(CONTRACT_CORPUS["base_request"])
            for change in case["changes"]
                parent = request
                for key in change["path"][1:end-1]
                    parent = parent[key isa Integer ? key + 1 : key]
                end
                key = last(change["path"])
                key = key isa Integer ? key + 1 : key
                if get(change, "remove", false)
                    delete!(parent, key)
                else
                    parent[key] = change["value"]
                end
            end
            if case["valid"]
                @test validate_system_request(request) === nothing
            else
                @test_throws ErrorException validate_system_request(request)
            end
        end
    end
    for value in (NaN, Inf)
        request = deepcopy(CONTRACT_CORPUS["base_request"])
        request["solver_options"]["custom"] = value
        @test_throws ErrorException validate_system_request(request)
    end
end

@testset "BEAT identity without Git" begin
    mktempdir() do directory
        source = joinpath(@__DIR__, "..", "src", "BeatEngineProvenance.jl")
        # Keep the copied engine and contract together in this isolated fixture.
        fixture_root = joinpath(directory, "engine")
        mkpath(joinpath(fixture_root, "src"))
        cp(source, joinpath(fixture_root, "src", "BeatEngineProvenance.jl"))
        mkpath(joinpath(directory, "beat_contract"))
        first_module = Module(:FirstProvenanceFixture)
        Base.include(first_module, joinpath(fixture_root, "src", "BeatEngineProvenance.jl"))
        first = withenv("PATH" => "") do
            Base.invokelatest(() -> first_module.BeatEngineProvenance.engine_identity())
        end
        @test first["repository_revision"] === nothing
        @test first["repository_dirty"] === nothing
        @test length(first["source_sha256"]) == 64
        write(joinpath(fixture_root, "solver.jl"), "# changed engine source\n")
        @test Base.invokelatest(() -> first_module.BeatEngineProvenance.engine_identity())["source_sha256"] == first["source_sha256"]
        second_module = Module(:SecondProvenanceFixture)
        Base.include(second_module, joinpath(fixture_root, "src", "BeatEngineProvenance.jl"))
        second = Base.invokelatest(() -> second_module.BeatEngineProvenance.engine_identity())
        @test second["source_sha256"] != first["source_sha256"]
    end
end

@testset "BEAT worker negotiation" begin
    info = worker_ready(Dict("cpu" => Dict("available" => true, "reason" => "")))
    @test info["protocol"]["version"] == 1
    @test info["contracts"]["system_request"] == [1]
    @test info["contracts"]["compiled_system"] == [1]
    @test info["contracts"]["system_result"] == [2]
    @test info["runtime"]["julia_version"] == string(VERSION)
    @test length(info["engine"]["source_sha256"]) == 64
    @test haskey(info["engine"]["source_files_sha256"], "julia_local/coupled_solver.jl")
    @test haskey(info["engine"], "repository_revision")
    @test haskey(info["engine"], "repository_dirty")
    @test length(info["runtime"]["project_sha256"]) == 64
    @test info["runtime"]["julia_threads"] >= 1
    @test info["runtime"]["blas_threads"] >= 1
    empty!(info["engine"]["source_files_sha256"])
    @test !isempty(worker_ready(Dict())["engine"]["source_files_sha256"])
    command = Dict{String,Any}("protocol_version" => 1, "operation" => "solve",
        "request" => "does-not-exist.json", "result_schema_version" => 2)
    @test validate_worker_submission(command) === nothing
    for value in (nothing, true, 1.0, 2, "1")
        bad = merge(command, Dict("protocol_version" => value))
        @test_throws ErrorException validate_worker_submission(bad)
    end
    for value in (nothing, true, 1, 3)
        @test_throws ErrorException validate_worker_submission(merge(command, Dict("result_schema_version" => value)))
    end
    @test_throws ErrorException validate_worker_submission(merge(command, Dict("operation" => "unknown")))
    @test_throws ErrorException validate_worker_submission(merge(command, Dict("request" => "")))
    field = Dict("protocol_version" => 1, "operation" => "bem_field", "request" => "field.json", "field_array_schema_version" => 1)
    @test validate_worker_submission(field) === nothing
    @test_throws ErrorException validate_worker_submission(merge(field, Dict("field_array_schema_version" => 2)))
end

# The worker's request parser for interface transfer-impedance layers, end to end through
# `solve_request`. The solver script is loaded up to its entry point (as memory_mesh_tests.jl
# does) into its own module, so its copies of the engine modules never replace the ones the
# rest of the suite uses.
@testset "interface transfer_impedance on the wire" begin
    solver_path = normpath(joinpath(@__DIR__, "..", "coupled_solver.jl"))
    solver = Core.eval(Main, :(module TransferImpedanceWireSolver end))
    Base.include_string(solver, first(split(read(solver_path, String), "\nif \"--worker\" in ARGS")), solver_path)
    fixtures = joinpath(@__DIR__, "fixtures")
    group(mesh_id, tag) = Dict{String,Any}("mesh_id" => mesh_id, "dimension" => 2, "tag" => tag, "name" => nothing)
    boundary(id, kind, region, mesh_id, tag) = Dict{String,Any}("id" => id, "name" => id, "kind" => kind,
        "region_id" => region, "group" => group(mesh_id, tag), "parameters" => Dict{String,Any}())
    felt(; extra...) = merge(Dict{String,Any}("model" => "resistance_mass",
        "resistance_pa_s_per_m" => 125.0, "mass_kg_per_m2" => 0.004), Dict{String,Any}(String(k) => v for (k, v) in extra))
    function request(; convention="exp(-i omega t)")
        system = Dict{String,Any}("id" => "system:transfer-impedance", "name" => "transfer impedance", "contract_version" => 1,
            "meshes" => [
                Dict{String,Any}("id" => "mesh:exterior", "name" => "Exterior", "file" => joinpath(fixtures, "exterior_conforming.msh"),
                    "purpose" => "bem_surface", "scale_to_m" => 0.001, "translation_m" => [0, 0, 0]),
                Dict{String,Any}("id" => "mesh:interior", "name" => "Interior", "file" => joinpath(fixtures, "femvolume.msh"),
                    "purpose" => "fem_volume", "scale_to_m" => 0.001, "translation_m" => [0, 0, 0]),
            ],
            "regions" => [
                Dict{String,Any}("id" => "region:exterior", "name" => "Exterior", "kind" => "unbounded_air", "mesh_ids" => ["mesh:exterior"],
                    "volume_groups" => [], "sound_speed_m_per_s" => 343.0, "density_kg_per_m3" => 1.21, "loss_model" => Dict()),
                Dict{String,Any}("id" => "region:interior", "name" => "Interior", "kind" => "bounded_air", "mesh_ids" => ["mesh:interior"],
                    "volume_groups" => [Dict("mesh_id" => "mesh:interior", "dimension" => 3, "tag" => 1, "name" => nothing)],
                    "sound_speed_m_per_s" => 343.0, "density_kg_per_m3" => 1.21, "loss_model" => Dict()),
            ],
            "boundaries" => [
                boundary("boundary:radiator", "moving", "region:interior", "mesh:interior", 2),
                boundary("boundary:interior-interface", "interface", "region:interior", "mesh:interior", 3),
                boundary("boundary:walls", "rigid", "region:interior", "mesh:interior", 4),
                boundary("boundary:exterior-box", "rigid", "region:exterior", "mesh:exterior", 1),
                boundary("boundary:exterior-interface", "interface", "region:exterior", "mesh:exterior", 2),
            ],
            "interfaces" => [Dict{String,Any}("id" => "interface:one", "name" => "One",
                "bounded_boundary_id" => "boundary:interior-interface", "unbounded_boundary_id" => "boundary:exterior-interface",
                "topology" => Dict("fem_vertex_indices" => [0], "fem_to_bem_vertex_indices" => [0], "fem_face_indices" => [0],
                    "bem_face_indices" => [0], "normal_sign" => [-1], "max_coordinate_error" => 0.0,
                    "fem_facets_on_tetra_boundary" => 1, "bem_boundary_edges" => 0))],
            "components" => [Dict{String,Any}("id" => "component:radiator", "name" => "Radiator", "kind" => "ideal_velocity_source",
                "boundary_ids" => ["boundary:radiator"], "parameters" => Dict())],
            "excitation_ports" => [Dict{String,Any}("id" => "excitation:radiator", "name" => "Radiator",
                "component_id" => "component:radiator", "kind" => "normal_velocity")],
        )
        return Dict{String,Any}("schema_version" => 1, "compiled_system" => system, "frequencies_hz" => [500.0],
            "excitation_port_ids" => ["excitation:radiator"],
            "outputs" => [Dict{String,Any}("id" => "velocity", "quantity" => "interface_average_normal_velocity",
                "target_ids" => [], "options" => Dict())],
            "solver_options" => Dict{String,Any}("precision" => "float64", "bem_backend" => "cpu", "phasor_convention" => convention))
    end
    boundary_parameters(request, id) =
        only(b for b in request["compiled_system"]["boundaries"] if b["id"] == id)["parameters"]
    function solve(request)
        events = mktemp() do _, stream
            redirect_stdout(stream) do
                Base.invokelatest(solver.solve_request, request)
            end
            flush(stream)
            seekstart(stream)
            [JSON.parse(line) for line in eachline(stream) if startswith(line, "{")]
        end
        return only(event for event in events if haskey(event, "diagnostics"))["diagnostics"]
    end
    refused(message, request) = @test_throws message Base.invokelatest(solver.solve_request, request)

    @testset "the layer is parsed and reported with the convention's sign" begin
        @test !haskey(solve(request()), "interface_transfer_impedance")
        omega = 2pi * 500.0
        for (convention, mass_reactance) in (("exp(-i omega t)", -omega * 0.004), ("exp(+i omega t)", omega * 0.004))
            layered = request(; convention=convention)
            boundary_parameters(layered, "boundary:interior-interface")["transfer_impedance"] = felt()
            diagnostics = solve(layered)
            record = diagnostics["interface_transfer_impedance"]
            @test record["phasor_convention"] == convention
            @test record["normal_orientation"] == "bounded_region_outward"
            layer = only(record["layers"])
            @test layer["interface_id"] == "interface:one"
            @test layer["boundary_id"] == "boundary:interior-interface"
            @test layer["resistance_pa_s_per_m"] == 125.0
            @test layer["mass_kg_per_m2"] == 0.004
            @test layer["impedance_real_pa_s_per_m"] == 125.0
            @test layer["impedance_imag_pa_s_per_m"] ≈ mass_reactance rtol = 1e-12
            # The residual of p_F - p_B = Z_s v_n, not of plain continuity.
            @test only(diagnostics["interface_pressure_continuity_errors"]) < 1e-8
        end
        # An exactly-zero layer is recorded but leaves the interface unmodified.
        zero_layer = request()
        boundary_parameters(zero_layer, "boundary:interior-interface")["transfer_impedance"] =
            felt(; resistance_pa_s_per_m=0, mass_kg_per_m2=0.0)
        @test only(solve(zero_layer)["interface_transfer_impedance"]["layers"])["impedance_imag_pa_s_per_m"] == 0
    end

    @testset "malformed and misplaced layers are refused" begin
        for (layer, message) in (
            (felt(; damping=1.0), "has unsupported keys: damping"),
            (delete!(felt(), "mass_kg_per_m2"), "is missing: mass_kg_per_m2"),
            (felt(; model="table"), "has unsupported model \"table\""),
            (felt(; resistance_pa_s_per_m=-1.0), "resistance_pa_s_per_m must be finite and non-negative"),
            (felt(; mass_kg_per_m2=-0.001), "mass_kg_per_m2 must be finite and non-negative"),
            (felt(; resistance_pa_s_per_m=true), "resistance_pa_s_per_m must be a real number"),
            (felt(; mass_kg_per_m2="0.004"), "mass_kg_per_m2 must be a real number"),
            (125.0, "must be an object"),
        )
            bad = request()
            boundary_parameters(bad, "boundary:interior-interface")["transfer_impedance"] = layer
            refused(message, bad)
        end
        for (boundary_id, message) in (
            ("boundary:walls", "requires kind \"interface\"; got \"rigid\""),
            ("boundary:exterior-interface", "the layer belongs on the interface's bounded (FEM-side) boundary"),
        )
            misplaced = request()
            boundary_parameters(misplaced, boundary_id)["transfer_impedance"] = felt()
            refused(message, misplaced)
        end
        # An interface-kind boundary that no interface record names.
        orphan = request()
        walls = only(b for b in orphan["compiled_system"]["boundaries"] if b["id"] == "boundary:walls")
        walls["kind"] = "interface"
        walls["parameters"]["transfer_impedance"] = felt()
        refused("\"boundary:walls\": the boundary is not the bounded boundary of any interface", orphan)
        # The contract refuses a non-finite value before the parser sees it.
        nonfinite = request()
        boundary_parameters(nonfinite, "boundary:interior-interface")["transfer_impedance"] =
            felt(; resistance_pa_s_per_m=Inf)
        @test_throws ErrorException Base.invokelatest(solver.solve_request, nonfinite)
    end

    @testset "the speaker ROM refuses a layered system" begin
        fem_mesh = solver.load_gmsh41_volume(joinpath(fixtures, "femvolume.msh"), 0.001)
        bem_mesh = solver.load_gmsh22_with_tags(joinpath(fixtures, "exterior_conforming.msh"), 0.001)
        interface_map = solver.build_conforming_interface_map(fem_mesh, bem_mesh, solver.physical_tag(fem_mesh, 2, "Interface"), 2)
        system = Base.invokelatest(solver.build_condensed_coupled_system, fem_mesh, bem_mesh, interface_map, 500.0, 343.0, 1.21;
            quadrature_order=1, singular_order=1,
            interface_transfer_impedances=[(interface_id="interface:one", boundary_id="boundary:interior-interface",
                interface_dofs=1:length(interface_map.fem_vertex_indices), resistance_pa_s_per_m=125.0, mass_kg_per_m2=0.0)])
        try
            @test_throws "Speaker ROM construction cannot hold interface transfer_impedance (interfaces \"interface:one\")" Base.invokelatest(
                solver.speaker_interior_state_matrix, system)
        finally
            Base.invokelatest(solver.release_condensed_coupled_system!, system)
        end
    end
end
