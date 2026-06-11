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
        layout = euler(Dict("A" => 5.0, "B" => 3.0, "A&B" => 1.0); seed=1)
        @test layout.shape == "circle"
        @test length(layout.shapes) == 2
        @test layout.shapes[1].type == "circle"
        @test layout.shapes[1].radius > 0
        @test layout.metrics.loss >= 0
        # every shape carries a label anchor for label placement
        @test haskey(layout.shapes[1], :label_anchor)
    end

    @testset "euler ellipses with complement" begin
        layout = euler(Dict("A" => 4.0, "B" => 2.0, "A&B" => 1.0);
                       shape="ellipse", complement=3.0, seed=2)
        @test layout.shape == "ellipse"
        @test layout.shapes[1].type == "ellipse"
        @test haskey(layout, :container)
    end

    @testset "venn ellipses" begin
        layout = venn(["A", "B", "C"]; shape="ellipse")
        @test length(layout.shapes) == 3
        @test layout.shapes[1].type == "ellipse"
    end

    @testset "errors surface, don't crash" begin
        @test_throws ErrorException euler(Dict("A" => 1.0); shape="hexagon")
    end
end
