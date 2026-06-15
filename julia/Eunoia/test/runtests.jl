using Test

# The native library must be locatable. In CI / local dev without a published
# artifact, build the cdylib and point EUNOIA_CAPI_LIB at it before loading the
# module.
if !haskey(ENV, "EUNOIA_CAPI_LIB") &&
   !isfile(joinpath(@__DIR__, "..", "Artifacts.toml"))
    repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    @info "building eunoia-capi for tests"
    run(`cargo build -p eunoia-capi --release`)
    ext = Sys.iswindows() ? "dll" : (Sys.isapple() ? "dylib" : "so")
    stem = Sys.iswindows() ? "eunoia_capi" : "libeunoia_capi"
    ENV["EUNOIA_CAPI_LIB"] = joinpath(repo_root, "target", "release", "$stem.$ext")
end

using Eunoia

@testset "Eunoia" begin
    @testset "version" begin
        @test !isempty(version())
    end

    @testset "euler circles" begin
        fit = euler(Dict("A" => 5.0, "B" => 3.0, "A&B" => 1.0); seed=1)
        @test fit isa EulerFit
        @test length(fit.shapes) == 2
        @test fit.shapes[1] isa Circle
        @test fit.shapes[1].radius > 0
        @test fit.loss >= 0
        # every shape carries a label anchor for label placement
        @test fit.shapes[1].label_anchor isa Point
        @test haskey(fit.original_values, "A")
        @test haskey(fit.fitted_values, "A")
        @test fit.residuals["A&B"] ≈
              fit.original_values["A&B"] - fit.fitted_values["A&B"]
        @test fit.container === nothing
    end

    @testset "euler ellipses with complement" begin
        fit = euler(Dict("A" => 4.0, "B" => 2.0, "A&B" => 1.0);
                    shape="ellipse", complement=3.0, seed=2)
        @test fit.shapes[1] isa Ellipse
        @test fit.container isa Container
        @test fit.container.width > 0
    end

    @testset "venn ellipses" begin
        fit = venn(["A", "B", "C"]; shape="ellipse")
        @test fit isa VennFit
        @test length(fit.shapes) == 3
        @test fit.shapes[1] isa Ellipse
    end

    @testset "membership-list input" begin
        # x→{A}, y→{A,B}, z→{A,B}, w→{B}
        fit = euler(Dict("A" => ["x", "y", "z"], "B" => ["y", "z", "w"]); seed=1)
        @test fit isa EulerFit
        @test length(fit.shapes) == 2
        @test fit.original_values["A"] == 1.0
        @test fit.original_values["B"] == 1.0
        @test fit.original_values["A&B"] == 2.0
    end

    @testset "inclusive input reconstruction" begin
        fit = euler(Dict("A" => 13.0, "B" => 8.0, "A&B" => 3.0);
                    input_type="inclusive", seed=1)
        # original_values stay in the user's (inclusive) scale
        @test fit.original_values["A"] == 13.0
        @test fit.original_values["A&B"] == 3.0
        # a 2-set circle case fits exactly, so fitted ≈ original in that scale
        @test isapprox(fit.fitted_values["A"], 13.0; atol=1e-6)
        @test isapprox(fit.fitted_values["B"], 8.0; atol=1e-6)
        @test isapprox(fit.fitted_values["A&B"], 3.0; atol=1e-6)
    end

    @testset "venn input forms" begin
        by_int = venn(3; shape="ellipse")
        @test [s.set for s in by_int.shapes] == ["A", "B", "C"]

        by_map = venn(Dict("A&B" => 1.0, "C" => 1.0); shape="ellipse")
        @test Set(s.set for s in by_map.shapes) == Set(["A", "B", "C"])
    end

    @testset "input errors" begin
        # membership + inclusive is contradictory
        @test_throws ErrorException euler(
            Dict("A" => ["x"], "B" => ["y"]); input_type="inclusive")
        # mixed membership/area values are ambiguous
        @test_throws ErrorException euler(Dict("A" => ["x"], "B" => 2.0))
        # venn rejects a bare string and a Bool
        @test_throws ArgumentError venn("ABC")
        @test_throws ArgumentError venn(true)
    end

    @testset "region_error and plot_data" begin
        fit = euler(Dict("A" => 5.0, "B" => 3.0, "A&B" => 1.0); seed=1)
        @test haskey(fit.region_error, "A&B")
        @test fit.region_error["A&B"] >= 0
        # plot_data carries renderable geometry for plotting
        @test haskey(fit.plot_data, :region_pieces)
        @test haskey(fit.plot_data.region_pieces, Symbol("A&B"))
        @test haskey(fit.plot_data, :shape_outlines)
        @test haskey(fit.plot_data, :set_anchors)
    end

    @testset "fitting knobs" begin
        base = Dict("A" => 5.0, "B" => 3.0, "A&B" => 1.0)

        # Loss + numeric knobs are accepted and produce a populated fit.
        fit = euler(base; seed=1, loss="sum_absolute", n_restarts=3,
                    max_iterations=50, tolerance=1e-4, jobs=1)
        @test fit isa EulerFit
        @test length(fit.shapes) == 2
        @test fit.loss >= 0

        # Smooth loss with an explicit eps.
        @test euler(base; seed=1, loss="smooth_diag_error", loss_eps=0.01) isa
              EulerFit

        # Solver / sampler knobs.
        @test euler(base; seed=1, optimizer="levenberg_marquardt",
                    mds_solver="lbfgs", initial_sampler="latin_hypercube") isa
              EulerFit
    end

    @testset "plot knobs" begin
        base = Dict("A" => 5.0, "B" => 3.0, "A&B" => 1.0)

        # `n_vertices` is observable end-to-end: it sets how densely each set's
        # outline is polygonized in `plot_data.shape_outlines`.
        coarse = euler(base; seed=1, n_vertices=40, label_precision=0.05,
                       sliver_threshold=1e-2)
        @test coarse isa EulerFit
        default = euler(base; seed=1)
        @test length(coarse.plot_data.shape_outlines.A) <
              length(default.plot_data.shape_outlines.A)
        @test 30 <= length(coarse.plot_data.shape_outlines.A) <= 60
    end

    @testset "label placement" begin
        base = Dict("A" => 5.0, "B" => 3.0, "A&B" => 1.0)
        fit = euler(base; seed=1)

        # Tiny labels fit inside every region: interior placement, no leader.
        small = place_labels(fit, Dict("A" => (0.1, 0.1), "B" => (0.1, 0.1),
                                       "A&B" => (0.05, 0.05)))
        @test small isa Dict{String,LabelPlacement}
        for combo in ("A", "B", "A&B")
            @test haskey(small, combo)
            @test small[combo].kind === :interior
            @test small[combo].tether === nothing
            @test small[combo].anchor isa Eunoia.Point
        end

        # An oversized label can't fit, so it is pushed outside with a leader.
        big = place_labels(fit, Dict("A&B" => (10.0, 10.0));
                           placement="force_directed")
        p = big["A&B"]
        @test startswith(String(p.kind), "exterior_")
        @test p.tether isa Eunoia.Point
        @test p.leader_end isa Eunoia.Point

        # Bad enum tokens are rejected by the native core.
        @test_throws ErrorException place_labels(fit, Dict("A" => (0.1, 0.1));
                                                 placement="bogus")
        @test_throws ErrorException place_labels(fit, Dict("A" => (0.1, 0.1));
                                                 tether="middle")
    end

    @testset "show" begin
        fit = euler(Dict("A" => 5.0, "B" => 3.0, "A&B" => 1.0); seed=1)
        str = sprint(show, MIME("text/plain"), fit)
        @test occursin("EulerFit", str)
        @test occursin("original", str)
        @test occursin("fitted", str)
        @test occursin("regionError", str)

        vstr = sprint(show, MIME("text/plain"), venn(["A", "B", "C"]))
        @test occursin("VennFit", vstr)
    end

    @testset "errors surface, don't crash" begin
        @test_throws ErrorException euler(Dict("A" => 1.0); shape="hexagon")
        # Invalid enum knobs are rejected by the native core.
        @test_throws ErrorException euler(Dict("A" => 1.0); loss="frobnicate")
        @test_throws ErrorException euler(Dict("A" => 1.0); optimizer="genetic")
        @test_throws ErrorException euler(Dict("A" => 1.0); mds_solver="gauss")
        @test_throws ErrorException euler(Dict("A" => 1.0); initial_sampler="sobol")
    end

    @testset "plotting stubs error without a backend" begin
        fit = euler(Dict("A" => 5.0, "B" => 3.0, "A&B" => 1.0); seed=1)
        # Until a Makie backend triggers the extension, the stubs explain how to
        # enable plotting instead of throwing a MethodError.
        if Base.get_extension(Eunoia, :EunoiaMakieExt) === nothing
            @test_throws ErrorException eunoiaplot(fit)
        end
    end
end

# ---------------------------------------------------------------------------
# Makie extension — opt-in (heavy precompile). Runs only under
# EUNOIA_TEST_MAKIE=true in the dedicated test/makie environment.
# ---------------------------------------------------------------------------
if get(ENV, "EUNOIA_TEST_MAKIE", "false") in ("true", "1")
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "makie"))
    Pkg.develop(path=normpath(joinpath(@__DIR__, "..")))
    Pkg.instantiate()
    using CairoMakie
    const MK = CairoMakie.Makie

    # Direct children of the recipe plot are exactly the primitives we draw
    # (one poly!/lines!/text! call each); their own internal sub-plots live
    # deeper, so we count at depth 1.
    npoly(p) = count(x -> x isa MK.Poly, p.plots)
    nlines(p) = count(x -> x isa MK.Lines, p.plots)
    # A single-position text!(...) stores its string as a 1-element vector, so
    # flatten each Text plot's `text` observable into the running list.
    function texts(p)
        out = String[]
        for x in p.plots
            x isa MK.Text || continue
            v = x.text[]
            v isa AbstractString && (push!(out, v); continue)
            v isa AbstractVector && append!(out, (s for s in v if s isa AbstractString))
        end
        return out
    end

    @testset "Makie extension" begin
        @test Base.get_extension(Eunoia, :EunoiaMakieExt) !== nothing

        fit = euler(Dict("A" => 10.0, "B" => 5.0, "A&B" => 3.0); seed=1)

        @testset "figure + primitives" begin
            fap = eunoiaplot(fit)
            @test fap.figure isa Figure
            @test fap.axis isa Axis
            @test fap.axis.aspect[] isa DataAspect
            @test fap.axis.xticksvisible[] == false
            @test fap.axis.bottomspinevisible[] == false
            p = fap.plot
            @test npoly(p) >= 3            # A-only, B-only, A&B region fills
            @test nlines(p) >= 2           # one outline per set
            @test "A" in texts(p) && "B" in texts(p)
            @test (MK.colorbuffer(fap.figure); true)   # headless render smoke test
        end

        @testset "labels off" begin
            t = texts(eunoiaplot(fit; labels=false).plot)
            @test !("A" in t) && !("B" in t)
        end

        @testset "labels per-set replacement" begin
            t = texts(eunoiaplot(fit; labels=Dict("A" => "Group A")).plot)
            @test "Group A" in t
            @test "B" in t
            @test !("A" in t)              # A replaced
        end

        @testset "quantities" begin
            t = texts(eunoiaplot(fit; quantities=true).plot)
            @test any(s -> occursin("10", s), t)        # original count of A
            tp = texts(eunoiaplot(fit; quantities="percent").plot)
            @test any(s -> endswith(s, "%"), tp)
        end

        @testset "legend hides inline labels + adds a Legend block" begin
            fap = eunoiaplot(fit; legend=true)
            @test any(c -> c isa Legend, fap.figure.content)
            @test isempty(texts(fap.plot))              # inline labels default off
        end

        @testset "complement draws a container box" begin
            plain = eunoiaplot(fit).plot
            withc = eunoiaplot(euler(Dict("A" => 4.0, "B" => 2.0, "A&B" => 1.0);
                                     complement=3.0, seed=2)).plot
            @test npoly(withc) == npoly(plain) + 1
        end

        @testset "compose into an existing axis" begin
            fig = Figure()
            ax = Axis(fig[1, 1])
            p = eunoiaplot!(ax, fit)
            @test npoly(p) >= 3
        end

        @testset "ellipse / square / rectangle render" begin
            for shp in ("ellipse", "square", "rectangle")
                p = eunoiaplot(euler(Dict("A" => 5.0, "B" => 3.0, "A&B" => 1.0);
                                     shape=shp, seed=1)).plot
                @test npoly(p) >= 3
            end
        end

        @testset "venn renders" begin
            p = eunoiaplot(venn(["A", "B", "C"]; shape="ellipse")).plot
            @test npoly(p) >= 1
            @test (MK.colorbuffer(eunoiaplot(venn(3)).figure); true)
        end
    end
end
