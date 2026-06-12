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

    @testset "show" begin
        fit = euler(Dict("A" => 5.0, "B" => 3.0, "A&B" => 1.0); seed=1)
        str = sprint(show, MIME("text/plain"), fit)
        @test occursin("EulerFit", str)
        @test occursin("original", str)
        @test occursin("fitted", str)
        # regionError column is omitted until the native lib emits it
        @test !occursin("regionError", str)

        vstr = sprint(show, MIME("text/plain"), venn(["A", "B", "C"]))
        @test occursin("VennFit", vstr)
    end

    @testset "errors surface, don't crash" begin
        @test_throws ErrorException euler(Dict("A" => 1.0); shape="hexagon")
    end
end
