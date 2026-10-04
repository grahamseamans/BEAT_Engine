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

# The layer against physics and against the other formulations: voltage-driven transducers through
# every condensed layout, phasor conjugation with a mass term, the power balance, a lumped
# Helmholtz-resonator model, and single precision with a large resistance.
@testset "interface transfer impedance physics" begin
    fixture_mesh(::Type{T}) where {T} = begin
        fem = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), T(0.001))
        bem = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), T(0.001))
        (fem=fem, bem=bem, interface_map=build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2))
    end
    mesh64, mesh32 = fixture_mesh(Float64), fixture_mesh(Float32)
    fem_mesh, bem_mesh, interface_map = mesh64
    radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
    interface_count = length(interface_map.fem_vertex_indices)
    sound_speed, density = 343.0, 1.21
    characteristic_impedance = density * sound_speed
    resistance_pa_s_per_m, mass_kg_per_m2 = 125.0, 0.004
    elimination_switches = (
        "BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION",
        "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION",
        "BLAB_COUPLED_TRANSDUCER_CONDENSATION",
        "BLAB_COUPLED_DENSE_REFINEMENT",
    )
    transfer_layers(resistance, mass) = iszero(resistance) && iszero(mass) ? NamedTuple[] : [(
        interface_id="interface:test",
        boundary_id="boundary:test-bounded",
        interface_dofs=1:interface_count,
        resistance_pa_s_per_m=resistance,
        mass_kg_per_m2=mass,
    )]
    monolithic_system(frequency_hz, layers; mesh=mesh64, validation_diagnostics=true, extra...) = build_coupled_system(
        mesh.fem, mesh.bem, mesh.interface_map, frequency_hz, sound_speed, density;
        quadrature_order=CONDENSED_QUADRATURE_ORDER,
        singular_order=CONDENSED_SINGULAR_ORDER,
        validation_diagnostics=validation_diagnostics,
        bem_backend=:cpu,
        interface_transfer_impedances=layers,
        extra...,
    )
    condensed_system(frequency_hz, layers; mesh=mesh64, transducers=ElectrodynamicTransducer{Float64}[], switches...) =
        withenv(
            (name => nothing for name in elimination_switches)...,
            (String(name) => value for (name, value) in switches)...,
        ) do
            build_condensed_coupled_system(
                mesh.fem, mesh.bem, mesh.interface_map, frequency_hz, sound_speed, density;
                quadrature_order=CONDENSED_QUADRATURE_ORDER,
                singular_order=CONDENSED_SINGULAR_ORDER,
                transducers=transducers,
                interface_transfer_impedances=layers,
            )
        end
    release!(system) = system.formulation == :fem_interface_condensed ?
                       release_condensed_coupled_system!(system) : release_coupled_system!(system)
    solve_excitation(system, excitation) = system.formulation == :fem_interface_condensed ?
                                           only(solve_condensed_coupled_excitations(system, [excitation])) :
                                           only(solve_coupled_excitations(system, [excitation]))
    with_system(f, system) = try
        f(system)
    finally
        release!(system)
    end
    relative_error(reference, candidate) =
        norm(ComplexF64.(candidate) .- ComplexF64.(reference)) / norm(ComplexF64.(reference))
    radiator_drive = (kind=:normal_velocity, radiator_tag=radiator_tag, transducer_index=0, amplitude=ComplexF64(1))
    voltage_drive(amplitude) = (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(amplitude))
    # The transducer of the condensed-versus-monolithic gate (coupled_condensed_tests.jl).
    transducer = ElectrodynamicTransducer{Float64}(
        "component:test", [radiator_tag], [1.0], [1], [-1.0], SVector(0.0, 0.0, 1.0),
        2.0, 1, 6.0, 0.0005, 7.0, 0.015, 0.0005, 1.0,
    )
    transducer_fields = (:fem_pressure, :bem_pressure, :interface_flux, :bem_neumann, :diaphragm_velocity, :voice_coil_current)
    layer = transfer_layers(resistance_pa_s_per_m, mass_kg_per_m2)

    # Consistent P1 interface mass restricted to the interface nodes: B_ii = ∫ φ_i φ_j over Γ.
    interface_mass = assemble_interface_operators(fem_mesh, bem_mesh, interface_map).fem_load[
        interface_map.fem_vertex_indices, :]
    interface_node_area = vec(sum(interface_mass; dims=1))   # ∫ φ_j over Γ; sums to the face area
    interface_area = sum(interface_node_area)

    @testset "voltage-driven transducer: monolithic equals condensed :none and :pressure (FP64)" begin
        reference = with_system(s -> solve_excitation(s, voltage_drive(1)),
            monolithic_system(500.0, layer; transducers=[transducer], validation_diagnostics=false))
        for (switches, elimination, transducer_condensed) in (
            (NamedTuple(), :none, false),
            ((BLAB_COUPLED_TRANSDUCER_CONDENSATION="1",), :none, true),
            ((BLAB_COUPLED_TRANSDUCER_CONDENSATION="1", BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION="on"), :pressure, true),
        )
            with_system(condensed_system(500.0, layer; transducers=[transducer], switches...)) do system
                @test system.interface_elimination == elimination
                @test system.transducer_condensation == transducer_condensed
                candidate = solve_excitation(system, voltage_drive(1))
                for field in transducer_fields
                    @test relative_error(getproperty(reference, field), getproperty(candidate, field)) < 1e-9
                end
                @test candidate.pressure_continuity_error < 1e-8
                @test candidate.fem_interior_residual < 1e-10
            end
        end
    end

    @testset "positive- and negative-time solutions with a mass term are conjugates" begin
        for condensed in (false, true)
            solutions = map((NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)) do convention
                with_phasor_convention(convention) do
                    system = condensed ?
                             condensed_system(500.0, layer; transducers=[transducer],
                                 BLAB_COUPLED_TRANSDUCER_CONDENSATION="1",
                                 BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION="on") :
                             monolithic_system(500.0, layer; transducers=[transducer], validation_diagnostics=false)
                    with_system(system) do system
                        # Z_s = R + time_derivative(omega) M: the mass enters with the convention's sign.
                        @test only(system_interface_transfer_impedance(system)).impedance ≈
                              resistance_pa_s_per_m + time_derivative(2pi * 500.0) * mass_kg_per_m2
                        solve_excitation(system, voltage_drive(convention == NEGATIVE_TIME_PHASOR ? 0.3 + 0.7im : 0.3 - 0.7im))
                    end
                end
            end
            for field in transducer_fields
                @test isapprox(getproperty(solutions[2], field), conj.(getproperty(solutions[1], field)); rtol=1e-8, atol=1e-10)
            end
        end
    end

    # V-TI3. With a lossless interior (no bulk loss, rigid walls) the FEM rows give the discrete
    # identity Re(p^H A p) real => power in at the piston = power through Γ (FEM side), to round-off.
    # Splitting p_F = p_B + Z_s v at the nodes, the FEM-side power is P_rad (the BEM trace pressure
    # times the same flux) plus P_layer = ½ R ∫|v|². All powers use the consistent interface mass.
    @testset "power balance: piston = layer + radiated, layer = ½ R ∫|v|²" begin
        for (frequency_hz, build) in (
            (500.0, (f, l) -> monolithic_system(f, l)),
            (300.0, (f, l) -> condensed_system(f, l; BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION="on")),
        )
            omega = 2pi * frequency_hz
            scale = neumann_scale(density, omega)
            piston_flux = assemble_prescribed_velocity_load(fem_mesh, radiator_tag, density, omega, ComplexF64(1)) ./ scale
            powers = map((NamedTuple[], layer)) do layers
                solution = with_system(s -> solve_excitation(s, radiator_drive), build(frequency_hz, layers))
                velocity = solution.interface_flux ./ scale
                fem_trace = solution.fem_pressure[interface_map.fem_vertex_indices]
                bem_trace = solution.bem_pressure[interface_map.fem_to_bem_vertex_indices]
                flux_integral = interface_mass * velocity
                layer_resistance = isempty(layers) ? 0.0 : resistance_pa_s_per_m
                (
                    piston=-real(transpose(solution.fem_pressure) * conj(piston_flux)) / 2,
                    radiated=real(transpose(bem_trace) * conj(flux_integral)) / 2,
                    layer=real(transpose(fem_trace .- bem_trace) * conj(flux_integral)) / 2,
                    layer_resistive=layer_resistance * real(velocity' * flux_integral) / 2,
                    layer_nodal=real(sum(interface_node_area .* (fem_trace .- bem_trace) .* conj.(velocity))) / 2,
                    layer_nodal_resistive=layer_resistance * sum(interface_node_area .* abs2.(velocity)) / 2,
                )
            end
            plain, layered = powers
            @info "V-TI3 power balance (W)" frequency_hz plain layered
            @test plain.piston > 0 && plain.radiated > 0
            @test abs(plain.layer) < 1e-9 * plain.piston
            @test abs(plain.piston - plain.radiated) < 1e-9 * plain.piston
            @test layered.piston > 0 && layered.radiated > 0 && layered.layer > 0
            @test abs(layered.layer - layered.layer_resistive) < 1e-9 * layered.layer
            @test abs(layered.layer_nodal - layered.layer_nodal_resistive) < 1e-9 * layered.layer_nodal
            @test abs(layered.piston - (layered.layer + layered.radiated)) < 1e-9 * layered.piston
        end
    end

    # V-TI2. The fixture is a Helmholtz resonator: a piston, a 785 cm³ cavity and a 20 mm,
    # r = 19 mm neck that opens through the interface. Lumped two-port (each in the active
    # convention, jω -> time_derivative(ω)): the piston's inflow U_in splits between the cavity
    # compliance Y_C = jω V/(ρc²) (V the tetrahedral volume) and the vent branch Z_v, so
    # U_vent = U_in / (1 + Y_C Z_v). The layer adds Z_s/S in series: Z_v = Z_v0 + Z_s/S.
    # Z_v0 (neck mass, end corrections, radiation) is read once per frequency from the R = M = 0
    # run; every layered run is then a prediction.
    @testset "lumped Helmholtz-resonator model (V-TI2)" begin
        cavity_volume = sum(fem_mesh.tetrahedra) do tetrahedron
            a, b, c, d = (fem_mesh.vertices[index] for index in tetrahedron)
            abs(dot(b - a, cross(c - a, d - a))) / 6
        end
        vent_flow(solution, omega) = sum(interface_node_area .* solution.interface_flux) / neumann_scale(density, omega)
        piston_inflow(omega) = -sum(assemble_prescribed_velocity_load(fem_mesh, radiator_tag, density, omega, ComplexF64(1))) /
                               neumann_scale(density, omega)
        cavity_admittance(omega) = time_derivative(omega) * cavity_volume / (density * sound_speed^2)
        solved_vent_flow(frequency_hz, resistance, mass) = vent_flow(
            with_system(s -> solve_excitation(s, radiator_drive), condensed_system(frequency_hz, transfer_layers(resistance, mass))),
            2pi * frequency_hz,
        )

        # Tolerance: 1 % in |U_vent| and 0.5° at f <= 100 Hz. The lumped Y_C ignores the cavity's
        # distributed correction, which the R = 0 extraction cannot absorb because it is a shunt
        # term. Measured on this fixture it is a clean f² law, |ΔU/U| ≈ 4.7e-7 f² (0.47 % at
        # 100 Hz, 4.2 % at 300 Hz), the same for every R and M, so it is the cavity, not the layer.
        # A layer with the wrong sign or a wrong area scaling misses by tens of percent.
        worst_magnitude_error, worst_phase_error_deg = 0.0, 0.0
        for frequency_hz in (60.0, 100.0)
            omega = 2pi * frequency_hz
            inflow, admittance = piston_inflow(omega), cavity_admittance(omega)
            plain_vent_impedance = (inflow / solved_vent_flow(frequency_hz, 0.0, 0.0) - 1) / admittance
            for (resistance, mass) in (
                (0.3characteristic_impedance, 0.0),
                (characteristic_impedance, 0.0),
                (3characteristic_impedance, 0.0),
                (0.0, 0.02),
                (resistance_pa_s_per_m, mass_kg_per_m2),
            )
                layer_impedance = resistance + time_derivative(omega) * mass
                predicted = inflow / (1 + admittance * (plain_vent_impedance + layer_impedance / interface_area))
                solved = solved_vent_flow(frequency_hz, resistance, mass)
                magnitude_error = abs(abs(solved) / abs(predicted) - 1)
                phase_error_deg = abs(rad2deg(angle(solved / predicted)))
                worst_magnitude_error = max(worst_magnitude_error, magnitude_error)
                worst_phase_error_deg = max(worst_phase_error_deg, phase_error_deg)
                @test magnitude_error < 0.01
                @test phase_error_deg < 0.5
            end
        end
        @info "V-TI2 lumped vent flow, worst over 60/100 Hz and five layers" worst_magnitude_error worst_phase_error_deg

        # The Helmholtz frequency, where Re(U_in / U_vent) = 1 + Re(Y_C Z_v) crosses zero, found by
        # secant iteration in f² (the crossing is linear in f² for a lumped resonator). A mass layer
        # chosen to lower it by 30 % in the lumped model must lower it by 30 % within 2 %; a
        # wrong-signed mass would move it up. The residual is the same f²-law cavity correction
        # (C_eff differs by ~2 % between the two resonances, so f_H by ~1 %).
        function helmholtz_frequency(mass, low_hz, high_hz)
            crossing(frequency_hz) = real(piston_inflow(2pi * frequency_hz) / solved_vent_flow(frequency_hz, 0.0, mass))
            x0, x1 = low_hz^2, high_hz^2
            g0, g1 = crossing(low_hz), crossing(high_hz)
            for _ in 1:8
                x2 = x1 - g1 * (x1 - x0) / (g1 - g0)
                abs(sqrt(x2) - sqrt(x1)) < 1e-3 && return sqrt(x2)
                x0, g0, x1 = x1, g1, x2
                g1 = crossing(sqrt(x1))
            end
            error("Helmholtz-frequency secant did not converge (mass $mass kg/m²).")
        end
        plain_helmholtz_hz = helmholtz_frequency(0.0, 250.0, 350.0)
        # Analytic neck: c/(2π) sqrt(S/(V L_eff)) with L_eff = 20 mm + end corrections ≈ 300 Hz.
        @test 250 < plain_helmholtz_hz < 350
        cavity_compliance = cavity_volume / (density * sound_speed^2)
        vent_mass = 1 / ((2pi * plain_helmholtz_hz)^2 * cavity_compliance)       # kg/m⁴, lumped
        target_ratio = 0.7
        layer_mass = interface_area * vent_mass * (1 / target_ratio^2 - 1)          # kg/m², ≈ 0.06
        loaded_helmholtz_hz = helmholtz_frequency(layer_mass, 0.65plain_helmholtz_hz, 0.75plain_helmholtz_hz)
        @info "V-TI2 Helmholtz shift" plain_helmholtz_hz loaded_helmholtz_hz layer_mass predicted_hz = target_ratio * plain_helmholtz_hz
        @test abs(loaded_helmholtz_hz / (target_ratio * plain_helmholtz_hz) - 1) < 0.02
    end

    # Single precision with a large resistance. Z_s v = p_F - p_B is a difference of two nearly
    # equal pressures once R >> ρc (the cavity pressure stays O(1) while the radiated pressure
    # falls as 1/R), so a ComplexF32 dense solve of the unreduced layout loses p_B to
    # cancellation: its relative error grows ~10× per decade of R. The flux and the interior are
    # unaffected, and the interface diagnostics do not see it. :pressure elimination (p_Γ = p_B +
    # d q is substituted, no difference is formed) and Float64 refinement both hold to ~1e-5.
    # The resistance is never clamped; this records where each path degrades.
    @testset "FP32 with large R: where it degrades" begin
        degradation = NamedTuple[]
        for decade in (0, 1, 2, 3, 6)
            resistance = 10.0^decade * characteristic_impedance
            reference = with_system(s -> solve_excitation(s, radiator_drive),
                condensed_system(500.0, transfer_layers(resistance, 0.0)))
            for (path, switches) in (
                (:none, NamedTuple()),
                (:pressure, (BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION="on",)),
                (:none_refined, (BLAB_COUPLED_DENSE_REFINEMENT="1",)),
            )
                candidate = with_system(
                    s -> solve_excitation(s, radiator_drive),
                    withenv(() -> build_condensed_coupled_system(
                            mesh32.fem, mesh32.bem, mesh32.interface_map, 500.0f0, Float32(sound_speed), Float32(density);
                            quadrature_order=CONDENSED_QUADRATURE_ORDER,
                            singular_order=CONDENSED_SINGULAR_ORDER,
                            interface_transfer_impedances=transfer_layers(Float32(resistance), 0.0f0),
                        ),
                        (name => nothing for name in elimination_switches)...,
                        (String(name) => value for (name, value) in pairs(switches))...,
                    ),
                )
                errors = (
                    decade=decade,
                    path=path,
                    bem_pressure=relative_error(reference.bem_pressure, candidate.bem_pressure),
                    interface_flux=relative_error(reference.interface_flux, candidate.interface_flux),
                    fem_pressure=relative_error(reference.fem_pressure, candidate.fem_pressure),
                    pressure_continuity_error=candidate.pressure_continuity_error,
                )
                push!(degradation, errors)
                @test all(isfinite, candidate.bem_pressure) && all(isfinite, candidate.interface_flux)
                # The precision gate's 1e-4 holds everywhere on the flux and interior ...
                @test errors.interface_flux < 1e-4
                @test errors.fem_pressure < 1e-4
                # ... and on the exterior for :pressure, for refinement, and for :none up to R = 10 ρc.
                (path != :none || decade <= 1) && @test errors.bem_pressure < 1e-4
            end
        end
        @info "FP32 transfer-impedance degradation (relative error vs FP64, 500 Hz, R = 10^decade ρc):\n" *
              join((repr(row) for row in degradation), "\n")
    end
end

# The accelerator arms of the layer: CUDA's sparse scatter on the device-resident monolithic and
# cuDSS-condensed systems, and ROCm's. They need hardware; without it they skip as broken tests
# with a warning, never as a pass.
for (backend, switch, available) in (
    (:cuda, "BLAB_RUN_COUPLED_CUDA", () -> cuda_available()),
    (:rocm, "BLAB_RUN_COUPLED_ROCM", () -> rocm_available()),
)
    @testset "interface transfer impedance on $backend" begin
        if get(ENV, switch, "0") == "1" && available()
            fem_mesh = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), Float32(0.001))
            bem_mesh = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), Float32(0.001))
            interface_map = build_conforming_interface_map(fem_mesh, bem_mesh, physical_tag(fem_mesh, 2, "Interface"), 2)
            radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
            layers = [(
                interface_id="interface:test",
                boundary_id="boundary:test-bounded",
                interface_dofs=1:length(interface_map.fem_vertex_indices),
                resistance_pa_s_per_m=125.0f0,
                mass_kg_per_m2=0.004f0,
            )]
            build(device_backend, static_condensation) = build_coupled_system(
                fem_mesh, bem_mesh, interface_map, 500.0f0, 343.0f0, 1.21f0;
                quadrature_order=CONDENSED_QUADRATURE_ORDER,
                singular_order=CONDENSED_SINGULAR_ORDER,
                validation_diagnostics=false,
                bem_backend=device_backend,
                static_condensation=static_condensation,
                interface_transfer_impedances=layers,
            )
            relative_error(reference, candidate) =
                norm(ComplexF64.(candidate) .- ComplexF64.(reference)) / norm(ComplexF64.(reference))
            cpu_system = build(:cpu, false)
            reference = try
                only(solve_coupled_systems(cpu_system, [radiator_tag]))
            finally
                release_coupled_system!(cpu_system)
            end
            for static_condensation in (false, true)
                system = build(backend, static_condensation)
                try
                    solution = only(solve_coupled_systems(system, [radiator_tag]))
                    for field in (:fem_pressure, :bem_pressure, :interface_flux)
                        @test relative_error(getproperty(reference, field), getproperty(solution, field)) < 1e-4
                    end
                    jump = solution.fem_pressure[interface_map.fem_vertex_indices] .-
                           solution.bem_pressure[interface_map.fem_to_bem_vertex_indices]
                    impedance = only(system_interface_transfer_impedance(system)).impedance
                    @test relative_error(impedance .* solution.interface_flux ./ neumann_scale(1.21f0, Float32(2pi * 500)), jump) < 1e-3
                finally
                    release_coupled_system!(system)
                end
            end
        else
            @warn "Interface transfer impedance on $backend NOT validated: set $switch=1 on hardware with a functional $backend device."
            @test_skip "$backend unavailable or $switch unset; the transfer-impedance device path is unvalidated."
        end
    end
end
