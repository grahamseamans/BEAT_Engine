isdefined(@__MODULE__, :CONDENSED_FIXTURE_ROOT) ||
    include(joinpath(@__DIR__, "coupled_condensed_test_setup.jl"))

# The transfer-impedance layer is one diagonal block `-d` on the flux-flux rows, so that the
# interface rows read p_F - p_B - d q = 0 with d = Z_s / neumann_scale. These tests pin the
# block itself, the jump it produces, and that an absent or exactly-zero layer changes nothing.
@testset "interface transfer impedance block" begin
    fem_mesh = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    bem_mesh = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
    interface_map = build_conforming_interface_map(
        fem_mesh,
        bem_mesh,
        physical_tag(fem_mesh, 2, "Interface"),
        2,
    )
    radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
    interface_count = length(interface_map.fem_vertex_indices)
    frequency_hz, sound_speed, density = 500.0, 343.0, 1.21
    omega = 2pi * frequency_hz
    # About 0.3 rho c, the felt target, plus a small inertive part.
    resistance_pa_s_per_m, mass_kg_per_m2 = 125.0, 0.004
    elimination_switches = (
        "BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION",
        "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION",
        "BLAB_COUPLED_TRANSDUCER_CONDENSATION",
    )

    transfer_layers(resistance, mass; interface_dofs=1:interface_count) = [(
        interface_id="interface:test",
        boundary_id="boundary:test-bounded",
        interface_dofs=interface_dofs,
        resistance_pa_s_per_m=resistance,
        mass_kg_per_m2=mass,
    )]
    monolithic_system(layers) = build_coupled_system(
        fem_mesh,
        bem_mesh,
        interface_map,
        frequency_hz,
        sound_speed,
        density;
        quadrature_order=CONDENSED_QUADRATURE_ORDER,
        singular_order=CONDENSED_SINGULAR_ORDER,
        validation_diagnostics=true,
        bem_backend=:cpu,
        interface_transfer_impedances=layers,
    )
    condensed_system(layers; switches...) = withenv(
        (name => nothing for name in elimination_switches)...,
        (String(name) => value for (name, value) in switches)...,
    ) do
        build_condensed_coupled_system(
            fem_mesh,
            bem_mesh,
            interface_map,
            frequency_hz,
            sound_speed,
            density;
            quadrature_order=CONDENSED_QUADRATURE_ORDER,
            singular_order=CONDENSED_SINGULAR_ORDER,
            interface_transfer_impedances=layers,
        )
    end
    solve_monolithic(system) = only(solve_coupled_systems(system, [radiator_tag]))
    solve_condensed(system) = only(solve_condensed_coupled_systems(system, [radiator_tag]))
    same_solution(reference, candidate) =
        reference.fem_pressure == candidate.fem_pressure &&
        reference.bem_pressure == candidate.bem_pressure &&
        reference.interface_flux == candidate.interface_flux &&
        reference.bem_neumann == candidate.bem_neumann &&
        reference.pressure_continuity_error == candidate.pressure_continuity_error &&
        reference.flux_conservation_error == candidate.flux_conservation_error
    relative_error(reference, candidate) = norm(candidate .- reference) / norm(reference)
    interface_pressure_difference(solution) =
        solution.fem_pressure[interface_map.fem_vertex_indices] .-
        solution.bem_pressure[interface_map.fem_to_bem_vertex_indices]

    plain = monolithic_system(NamedTuple[])
    zero_layer = monolithic_system(transfer_layers(0.0, 0.0))
    layered = monolithic_system(transfer_layers(resistance_pa_s_per_m, mass_kg_per_m2))
    try
        @testset "absent and exactly-zero layers are the unmodified interface, bit for bit" begin
            @test isempty(zero_layer.interface_transfer_impedance)
            @test zero_layer.coupled == plain.coupled
            @test same_solution(solve_monolithic(plain), solve_monolithic(zero_layer))
            for switches in (NamedTuple(), (BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION="on",))
                condensed_plain = condensed_system(NamedTuple[]; switches...)
                condensed_zero = condensed_system(transfer_layers(0.0, 0.0); switches...)
                try
                    @test condensed_zero.interface_elimination == condensed_plain.interface_elimination
                    @test same_solution(solve_condensed(condensed_plain), solve_condensed(condensed_zero))
                finally
                    release_condensed_coupled_system!(condensed_plain)
                    release_condensed_coupled_system!(condensed_zero)
                end
            end
        end

        @testset "the layer is exactly -d on the flux diagonal and nothing else" begin
            impedance = resistance_pa_s_per_m + time_derivative(omega) * mass_kg_per_m2
            jump_coefficient = impedance / neumann_scale(density, omega)
            layer = only(layered.interface_transfer_impedance)
            @test layer.impedance == impedance
            @test layer.jump_coefficient == jump_coefficient
            difference = layered.coupled - plain.coupled
            flux_range = layered.flux_range
            @test difference[flux_range, flux_range] == Matrix(Diagonal(fill(-jump_coefficient, interface_count)))
            difference[flux_range, flux_range] .= 0
            @test iszero(difference)
        end

        @testset "p_F - p_B = Z_s v_n at the interface nodes" begin
            plain_solution = solve_monolithic(plain)
            layered_solution = solve_monolithic(layered)
            impedance = only(layered.interface_transfer_impedance).impedance
            normal_velocity = layered_solution.interface_flux ./ neumann_scale(density, omega)
            pressure_difference = interface_pressure_difference(layered_solution)
            @test relative_error(impedance .* normal_velocity, pressure_difference) < 1e-10
            @test layered_solution.relative_residual < 1e-12
            @test layered_solution.pressure_continuity_error < 1e-10
            @test layered_solution.flux_conservation_error < 1e-10
            # The layer is not a no-op: the plain interface has (numerically) no jump.
            @test norm(interface_pressure_difference(plain_solution)) <
                  1e-8 * norm(pressure_difference)
            @test relative_error(plain_solution.bem_pressure, layered_solution.bem_pressure) > 1e-3
        end

        @testset "condensed :none and :pressure carry the same layer" begin
            reference = solve_monolithic(layered)
            layers = transfer_layers(resistance_pa_s_per_m, mass_kg_per_m2)
            for (switches, elimination) in (
                (NamedTuple(), :none),
                ((BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION="on",), :pressure),
            )
                system = condensed_system(layers; switches...)
                try
                    @test system.interface_elimination == elimination
                    candidate = solve_condensed(system)
                    @test relative_error(reference.fem_pressure, candidate.fem_pressure) < 1e-9
                    @test relative_error(reference.bem_pressure, candidate.bem_pressure) < 1e-9
                    @test relative_error(reference.interface_flux, candidate.interface_flux) < 1e-9
                    @test candidate.pressure_continuity_error < 1e-8
                finally
                    release_condensed_coupled_system!(system)
                end
            end
        end

        @testset "flux elimination refuses under on and falls back under auto" begin
            layers = transfer_layers(resistance_pa_s_per_m, mass_kg_per_m2)
            @test_throws "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION=on cannot hold interface transfer_impedance" condensed_system(
                layers;
                BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION="on",
            )
            system = condensed_system(layers; BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION="auto")
            try
                @test system.interface_elimination == :pressure
                @test any(startswith("interface flux elimination not used"), system.optimization_fallback_reasons)
            finally
                release_condensed_coupled_system!(system)
            end
        end

        @testset "invalid layers are refused with the parameter name" begin
            @test_throws "resistance_pa_s_per_m must be finite and non-negative" active_interface_transfer_impedances(
                transfer_layers(-1.0, 0.0), interface_count, Float64,
            )
            @test_throws "mass_kg_per_m2 must be finite and non-negative" active_interface_transfer_impedances(
                transfer_layers(1.0, NaN), interface_count, Float64,
            )
            @test_throws "are not a non-empty range" active_interface_transfer_impedances(
                transfer_layers(1.0, 0.0; interface_dofs=1:(interface_count + 1)), interface_count, Float64,
            )
            @test_throws "interface_transfer_impedances do not match" build_coupled_system(
                fem_mesh, bem_mesh, interface_map, frequency_hz, sound_speed, density;
                quadrature_order=CONDENSED_QUADRATURE_ORDER,
                singular_order=CONDENSED_SINGULAR_ORDER,
                validation_diagnostics=true,
                bem_backend=:cpu,
                cache=plain.cache,
                interface_transfer_impedances=transfer_layers(resistance_pa_s_per_m, 0.0),
            )
        end
    finally
        release_coupled_system!(plain)
        release_coupled_system!(zero_layer)
        release_coupled_system!(layered)
    end
end
