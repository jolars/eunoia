# Regenerate `Artifacts.toml` from a published GitHub release.
#
# This is the Option A "roll your own JLL" step: instead of BinaryBuilder +
# Yggdrasil, the per-platform `libeunoia_capi` tarballs are built by the
# `.github/workflows/julia-artifacts.yml` matrix and attached to a
# `julia-v*` release. This script downloads each tarball, hashes it, and writes
# the matching lazy, platform-tagged entry into `Artifacts.toml`.
#
# Usage:
#
#   julia --project=julia/Eunoia julia/Eunoia/gen/generate_artifacts.jl julia-v0.18.0
#
# Requires `ArtifactUtils` (add it to your global/temp env, not the package):
#
#   julia -e 'import Pkg; Pkg.add("ArtifactUtils")'

using ArtifactUtils
using Base.BinaryPlatforms

const TAG = isempty(ARGS) ? error("pass the release tag, e.g. julia-v0.18.0") : ARGS[1]
const REPO = "https://github.com/jolars/eunoia"
const BASE = "$REPO/releases/download/$TAG"
const ARTIFACTS_TOML = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))

# (Julia platform, tarball basename). The tarball names mirror the CI matrix's
# `julia` triplet column.
const TARGETS = [
    (Platform("x86_64", "linux"; libc="glibc"), "libeunoia_capi-x86_64-linux-gnu.tar.gz"),
    (Platform("aarch64", "linux"; libc="glibc"), "libeunoia_capi-aarch64-linux-gnu.tar.gz"),
    (Platform("x86_64", "macos"), "libeunoia_capi-x86_64-apple-darwin.tar.gz"),
    (Platform("aarch64", "macos"), "libeunoia_capi-aarch64-apple-darwin.tar.gz"),
    (Platform("x86_64", "windows"), "libeunoia_capi-x86_64-w64-mingw32.tar.gz"),
]

for (i, (platform, tarball)) in enumerate(TARGETS)
    url = "$BASE/$tarball"
    @info "adding artifact" platform url
    add_artifact!(
        ARTIFACTS_TOML,
        "eunoia",
        url;
        platform=platform,
        lazy=true,            # download on first use, not at install time
        force=true,
        clear=(i == 1),       # wipe stale entries on the first call only
    )
end

@info "wrote $ARTIFACTS_TOML"
