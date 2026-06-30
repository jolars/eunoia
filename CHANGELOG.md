# Changelog

## [1.8.0](https://github.com/jolars/eunoia/compare/v1.7.0...v1.8.0) (2026-06-30)

### Features
- **plotting:** add matched exterior label placement ([`e76314b`](https://github.com/jolars/eunoia/commit/e76314b702919d8929deb539f0570d2ec244d7e4))

## [1.7.0](https://github.com/jolars/eunoia/compare/v1.6.0...v1.7.0) (2026-06-30)

### Features
- **capi:** expose mads/cmaes optimizers and venn complement ([`496b60c`](https://github.com/jolars/eunoia/commit/496b60c352288ce96872c0d4d2c2a8f024857fa7))
- **web:** add a legend entry for the complement ([`c56faea`](https://github.com/jolars/eunoia/commit/c56faea5c53b3fb5e631c7992a8f3bd773d5f400))

## [1.6.0](https://github.com/jolars/eunoia/compare/v1.5.0...v1.6.0) (2026-06-26)

### Features
- **fitter:** auto-reduce restarts for small smooth-loss fits ([`1a61437`](https://github.com/jolars/eunoia/commit/1a6143709dd4013bd85db49de2787257c7c38819))
- **fitter:** wire MADS poll-size floor to tolerance; add lever sweep ([`90636b9`](https://github.com/jolars/eunoia/commit/90636b9d7f083e846b4bb82410906440e9788620))
- **fitter:** add MADS optimizer + benchmark vs Nelder-Mead ([`04a144b`](https://github.com/jolars/eunoia/commit/04a144bfb0329afd7f3e2ef4043e1754a1e78a47))
- **venn:** add rotated rectangle shape for 4 set venn diagrams ([`322c1d9`](https://github.com/jolars/eunoia/commit/322c1d9c7cc24ec23464116c35d252c41a4b6912))
- **web:** add rotated rectangles ([`33b9250`](https://github.com/jolars/eunoia/commit/33b92504cbce2b5c0be25ef524bab9f4388fd30f))
- **shapes:** add RotatedRectangle shape with derivative-free fitting ([`254adbd`](https://github.com/jolars/eunoia/commit/254adbd84888ecc8e6812782e4a0e7f26f012a6b))

## [1.5.0](https://github.com/jolars/eunoia/compare/v1.4.0...v1.5.0) (2026-06-20)

### Features
- **web:** validate complement input with inline error ([`ccea349`](https://github.com/jolars/eunoia/commit/ccea349f1c24fdb7070489fc9f4267ac3f934ca2))

### Bug Fixes
- **plotting:** make RegionPolygons::iter() deterministic ([`6b7a2c3`](https://github.com/jolars/eunoia/commit/6b7a2c38f921d09557e7f671c7d0b872db60066a))

### Performance Improvements
- bump basin to v1 ([`515df92`](https://github.com/jolars/eunoia/commit/515df9268214197f957cda7cc7f9c72bf2aba68c))

## [1.4.0](https://github.com/jolars/eunoia/compare/v1.3.0...v1.4.0) (2026-06-16)

### Features
- **plotting:** normalize placement of sets-within-sets ([`2d3de4d`](https://github.com/jolars/eunoia/commit/2d3de4d8ad11ce2f95003347c0f7aa99c914d85f))
- **plotting:** center plot layout within complement container ([`07e3338`](https://github.com/jolars/eunoia/commit/07e3338eb7215185cb5a1d4828b4b19ac0962229))
- **capi:** expose set_anchor_regions in plot_data ([`d810d58`](https://github.com/jolars/eunoia/commit/d810d58d51d08098a89c64ead943f3433e70ea75))
- remove julia bindings from this repo ([`50ad043`](https://github.com/jolars/eunoia/commit/50ad04381592cc31663cafbc751b7dcef25d93e0))

## [1.3.0](https://github.com/jolars/eunoia/compare/v1.2.0...v1.3.0) (2026-06-15)

### Features
- **julia:** surface max_sets as an euler kwarg ([`835968d`](https://github.com/jolars/eunoia/commit/835968d16b169376a4ecb144fc3616894b8d6b48))
- **capi:** forward max_sets into DiagramSpecBuilder ([`d292a25`](https://github.com/jolars/eunoia/commit/d292a250d64fadc3cf7b4c3cbd8e8f08101dc2ee))
- **julia:** collision-aware Makie labels with leaders ([`105e08c`](https://github.com/jolars/eunoia/commit/105e08cf8f5f9be4f9bd07c8418be83935e6520b))
- **capi:** add eunoia_place_labels entry point; surface as Eunoia.place_labels ([`d84fc14`](https://github.com/jolars/eunoia/commit/d84fc142e40d31ca18b960408a483445e792b48e))
- **capi:** forward plot-tuning knobs; surface as euler kwargs in Eunoia.jl ([`563fd77`](https://github.com/jolars/eunoia/commit/563fd77ea0fc311a99a6a991d4100f2179a421c4))

### Bug Fixes
- **ci:** repair Julia artifact build (zig naming + pinned-toolchain target) ([`7771ed8`](https://github.com/jolars/eunoia/commit/7771ed8cf7d8f932198028b6a3688a06ff3bd0e1))

## [1.2.0](https://github.com/jolars/eunoia/compare/v1.1.0...v1.2.0) (2026-06-14)

### Features
- add self-contained browser bundle ([`770ad10`](https://github.com/jolars/eunoia/commit/770ad103072052ee87b7462ee93de351c4ad0998))
- **capi:** forward fitting knobs; surface as euler kwargs in Eunoia.jl ([`a3329e0`](https://github.com/jolars/eunoia/commit/a3329e0e529802c4c278088fd64e507532f17565))

## [1.1.0](https://github.com/jolars/eunoia/compare/v1.0.0...v1.1.0) (2026-06-12)

### Features
- **loss:** add LogSumAbsolute and SmoothLogSumAbsolute ([`2682050`](https://github.com/jolars/eunoia/commit/2682050bb6e1f2625f43852adf6bdec224493ad9)), closes [#96](https://github.com/jolars/eunoia/issues/96)
- **julia:** add Makie plotting extension (Phase 3) ([`1477969`](https://github.com/jolars/eunoia/commit/1477969909882bf49954a34808e1db8a51e56040))
- **capi:** emit region_error and plot_data; surface in Eunoia.jl ([`f20b420`](https://github.com/jolars/eunoia/commit/f20b420921b4b8f761740384b899d5c667d56832))
- **julia:** membership input, inclusive reconstruction, venn input forms ([`9757e45`](https://github.com/jolars/eunoia/commit/9757e45f547eccac70ebc9e32a6110522b8dc930))
- **julia:** typed result model and show for Eunoia.jl ([`dbaa746`](https://github.com/jolars/eunoia/commit/dbaa74615ca1d87fa0a0e26be32c928fcd355205))

## [1.0.0](https://github.com/jolars/eunoia/compare/v0.18.0...v1.0.0) (2026-06-12)

This is the first stable release for Eunoia and it includes a number of
breaking changes (see below). Eunoia uses semantic versioning, so this 1.0.0
release indicates that the API is now considered stable and future releases
will follow the semantic versioning rules for backward compatibility. Unless
there is a major version bump, all changes will be backward compatible with
this release.

### Breaking changes
- **geometry:** unify BoundingBox and bounds implementations ([`b0ef190`](https://github.com/jolars/eunoia/commit/b0ef19004322763c895e3c1c3f4ed1307f469902))
- make input/config enums #[non_exhaustive] ([`79d7d5f`](https://github.com/jolars/eunoia/commit/79d7d5f7e95157feefe974d9e91c18a36e43888d))
- **plotting:** non_exhaustive + fluent setters on config structs ([`f3e648e`](https://github.com/jolars/eunoia/commit/f3e648efcdfb538b43c8699055a0be1d9dc8683d))
- make some output-types `#[non_exhaustive]` ([`4d5f045`](https://github.com/jolars/eunoia/commit/4d5f0453a4bb7a1ec835ec3eec25c4c009fe8027))
- **api:** seal internal math/geometry plumbing before 1.0 ([`5067e89`](https://github.com/jolars/eunoia/commit/5067e89b2d19cddbb27e296e08537afb34dc8be6))
- **api:** harden public enums and naming for 1.0 ([`604dad3`](https://github.com/jolars/eunoia/commit/604dad37e480505c1eac2323d3dea929d6a92c18))
- replace default optimizer and add all variants ([`f0dcf77`](https://github.com/jolars/eunoia/commit/f0dcf77f8617bf4ffd41cd98f5b307d1cc42ff24))
- **ts:** drop support for raw wasm bindings ([`9022f22`](https://github.com/jolars/eunoia/commit/9022f22835380c667f8df6e72e637fe972d4d6f8))

### Features
- add WIP julia package and C ABI bindings ([`7e29b0f`](https://github.com/jolars/eunoia/commit/7e29b0ff4e9a4d7cc9abca007ba2dfed9378e409))
- add `restarts` as option ([`2764dad`](https://github.com/jolars/eunoia/commit/2764dad04015525ac542002dbba4e3cfa459fe99))
- **geometry:** unify BoundingBox and bounds implementations ([`b0ef190`](https://github.com/jolars/eunoia/commit/b0ef19004322763c895e3c1c3f4ed1307f469902))
- deprecate `sse`/`rmse` in favor of `sum_squared` etc ([`c930288`](https://github.com/jolars/eunoia/commit/c930288470eaa6b4335efb040e679b2821ac1b47))
- make input/config enums #[non_exhaustive] ([`79d7d5f`](https://github.com/jolars/eunoia/commit/79d7d5f7e95157feefe974d9e91c18a36e43888d))
- **plotting:** non_exhaustive + fluent setters on config structs ([`f3e648e`](https://github.com/jolars/eunoia/commit/f3e648efcdfb538b43c8699055a0be1d9dc8683d))
- make some output-types `#[non_exhaustive]` ([`4d5f045`](https://github.com/jolars/eunoia/commit/4d5f0453a4bb7a1ec835ec3eec25c4c009fe8027))
- **ts:** thread set_anchor_regions through the wasm/ts path ([`cee98b8`](https://github.com/jolars/eunoia/commit/cee98b86acf3c1ae3e0585afe47d43adf2dd4f4b)), refs [#88](https://github.com/jolars/eunoia/issues/88)
- **plotting:** expose set_anchor_regions on PlotData ([`8795f04`](https://github.com/jolars/eunoia/commit/8795f04b8876ff944022e2ea6c908b586b023fcb)), closes [#88](https://github.com/jolars/eunoia/issues/88)
- **fitter:** add internal EscapeSolver knob to benchmark memetic escapes ([`498f1a0`](https://github.com/jolars/eunoia/commit/498f1a0c1e4a9511301c193051f8d9518e977a66))
- replace default optimizer and add all variants ([`f0dcf77`](https://github.com/jolars/eunoia/commit/f0dcf77f8617bf4ffd41cd98f5b307d1cc42ff24))
- **ts:** drop support for raw wasm bindings ([`9022f22`](https://github.com/jolars/eunoia/commit/9022f22835380c667f8df6e72e637fe972d4d6f8))

## [0.18.0](https://github.com/jolars/eunoia/compare/v0.17.0...v0.18.0) (2026-06-03)

### Features
- **ts:** add headless @jolars/eunoia/svg serializer ([`d0fb43b`](https://github.com/jolars/eunoia/commit/d0fb43b99411fcbd321a083d93f6a6853ad143d1))

## [0.17.0](https://github.com/jolars/eunoia/compare/v0.16.1...v0.17.0) (2026-06-01)

### Features
- upgrade to basin 0.9.0 and drop MSRV to 1.87.0 ([`48e89a5`](https://github.com/jolars/eunoia/commit/48e89a5f27aac2fc63c84e5d3494d44858cb0657))

### Bug Fixes
- bump MSRV to 1.88.0 ([`36e2d41`](https://github.com/jolars/eunoia/commit/36e2d41bcf87ced7d63dfda834be4eb552e7d754))
## [0.16.1](https://github.com/jolars/eunoia/compare/v0.16.0...v0.16.1) (2026-05-28)

### Bug Fixes
- **fitter:** perturb initial ellipse rotation per restart ([`171786f`](https://github.com/jolars/eunoia/commit/171786fe2844c6b16611f6324638bf4849564020))
- **fitter:** race TRF from MDS init inside CmaEsTrf when LM stalls ([`f0b02c3`](https://github.com/jolars/eunoia/commit/f0b02c3d9565f63b6b8d19632ca119469011b441))
## [0.16.0](https://github.com/jolars/eunoia/compare/v0.15.0...v0.16.0) (2026-05-28)

### Breaking changes
- remove plotting feature (make it non-optionable) ([`4056952`](https://github.com/jolars/eunoia/commit/405695285a2061a89e7e90adeac0e2369da3b047))
- **fitter:** add box-constrained TRF optimizers, default to CmaEsTrf ([`8eda26d`](https://github.com/jolars/eunoia/commit/8eda26d12d510b6ebbdac6cf320cecf81101a7cf))
- **plotting:** replace curved leaders with edge-coupled leader strategy ([`848c739`](https://github.com/jolars/eunoia/commit/848c739bfdbba21280f43e9ae5a9b725dbff0a4b))
- add opt-in `parallel` feauture and jobs parameter ([`a42e8a0`](https://github.com/jolars/eunoia/commit/a42e8a0c24cb2c3d6ca594a32ab8be3c004bd30a))
- drop argmin + argmin-math, complete the basin migration ([`92c6acf`](https://github.com/jolars/eunoia/commit/92c6acf18f17228562070dbceb2e2cd4f4d9d5d7))
- raise MSRV to 1.91.1 and adopt edition 2024 ([`2f3d282`](https://github.com/jolars/eunoia/commit/2f3d2822e3e106446befd43860c162f4d40baf13))

### Features
- remove plotting feature (make it non-optionable) ([`4056952`](https://github.com/jolars/eunoia/commit/405695285a2061a89e7e90adeac0e2369da3b047))
- **fitter:** add box-constrained TRF optimizers, default to CmaEsTrf ([`8eda26d`](https://github.com/jolars/eunoia/commit/8eda26d12d510b6ebbdac6cf320cecf81101a7cf))
- **plotting:** add elbow labeling strategy ([`79881eb`](https://github.com/jolars/eunoia/commit/79881ebc1515fb9afae8891143aad69b683bbf82))
- **plotting:** replace curved leaders with edge-coupled leader strategy ([`848c739`](https://github.com/jolars/eunoia/commit/848c739bfdbba21280f43e9ae5a9b725dbff0a4b))
- **ts:** add venn interface for canonical venn diagrams ([`4bd1304`](https://github.com/jolars/eunoia/commit/4bd13048fa41f64318fb7d039312bdc217cb7f56))
- **plotting:** add curved leaders ([`33c160e`](https://github.com/jolars/eunoia/commit/33c160e7ed0f9fc6f016d413e67af9515bf83e30))
- add opt-in `parallel` feauture and jobs parameter ([`a42e8a0`](https://github.com/jolars/eunoia/commit/a42e8a0c24cb2c3d6ca594a32ab8be3c004bd30a))

### Bug Fixes
- **plotting:** keep gap-straddling labels clear of neighbouring shapes ([`becae27`](https://github.com/jolars/eunoia/commit/becae277b29c358ec2d3b959eab1b1a31d791eb8))
- **wasm:** fix wasm build failure ([`d720e2c`](https://github.com/jolars/eunoia/commit/d720e2c3c5b13d784d61df4c82f337a5cdde5b69))
- **ts:** set esModuleInterop to true ([`0de05d5`](https://github.com/jolars/eunoia/commit/0de05d533c85380612f05b3aaa125fc41cb3009f))

### Performance Improvements
- **fitter:** parallelize corpus-quality test fits across cores ([`9dc0785`](https://github.com/jolars/eunoia/commit/9dc0785d269c479bcff3a517288f18f32e535819))
- **fitter:** score LM/TRF residual over sparse mask set ([`0bdaf66`](https://github.com/jolars/eunoia/commit/0bdaf660613e70472cc1799171d383ae190900f1)), issue [#89](https://github.com/jolars/eunoia/issues/89)
## [0.15.0](https://github.com/jolars/eunoia/compare/v0.14.0...v0.15.0) (2026-05-12)

### Features
- **plotting:** expose `leader_gap` ([`33704a4`](https://github.com/jolars/eunoia/commit/33704a4787986aea46597235b25b9b126469d59f))
- **plotting:** expose `leader_end` ([`bcb11ce`](https://github.com/jolars/eunoia/commit/bcb11ce47ad91ef99bf1a0faf2321000968ffb68))
## [0.14.0](https://github.com/jolars/eunoia/compare/v0.13.0...v0.14.0) (2026-05-11)

### Breaking changes
- remove strict placement API ([`33c1bf0`](https://github.com/jolars/eunoia/commit/33c1bf00397e728224a12ebf1d097b1a77cb0d46))

### Features
- remove strict placement API ([`33c1bf0`](https://github.com/jolars/eunoia/commit/33c1bf00397e728224a12ebf1d097b1a77cb0d46))
- improve raycasting for label placement ([`b239021`](https://github.com/jolars/eunoia/commit/b2390219bcbc0eb7df7d7f1009bf35b1158610a2))
- add simple labeling api entrypoit ([`40ef7c7`](https://github.com/jolars/eunoia/commit/40ef7c7c1074522e38f2ec88da0e5afbab7fb10f))
- implement force-directed labeling strategy ([`988a3cf`](https://github.com/jolars/eunoia/commit/988a3cf95cb1c1a2128613dce5548da1c4e7eab1))
- add label placement strategies ([`e1a72ff`](https://github.com/jolars/eunoia/commit/e1a72ff1c35ee2bf9178caa075c7a6c39b9efcc8))
- rename `fit` to `euler` ([`8cd1e16`](https://github.com/jolars/eunoia/commit/8cd1e1663109f96320436dba2094d65d89ec01af))
## [0.13.0](https://github.com/jolars/eunoia/compare/v0.12.0...v0.13.0) (2026-05-07)

### Features
- **plotting:** provide plotting details for complement ([`9568246`](https://github.com/jolars/eunoia/commit/9568246e9a86e164448588b40309b2ea9c061867)), closes [#56](https://github.com/jolars/eunoia/issues/56)

### Bug Fixes
- **web:** surface errors probably ([`b16ef94`](https://github.com/jolars/eunoia/commit/b16ef94800ad9da4f45c33171248a37e9e8cc571))
## [0.12.0](https://github.com/jolars/eunoia/compare/v0.11.0...v0.12.0) (2026-05-06)

### Features
- **web:** add a landig page and unify web layout ([`7ff0559`](https://github.com/jolars/eunoia/commit/7ff0559cecf1ce5152a19a2c23df3af420290e37))
- support `complement` as a bounding box ([`9503454`](https://github.com/jolars/eunoia/commit/950345418eb3a862cf1bf2c055e67dec8d32d22b))
- add ellipse-bounding box gradient via forward-diff ([`1d400b0`](https://github.com/jolars/eunoia/commit/1d400b08587809cb58c34d7bf4545f01aeee5f53))
- support rectangles ([`5e3e1fb`](https://github.com/jolars/eunoia/commit/5e3e1fb99f0d6490bda268dfa30e02d6eeb48061))

### Bug Fixes
- **web:** enforce stable ordering of sets ([`2d5c749`](https://github.com/jolars/eunoia/commit/2d5c749e4f85b446c2c4cefa1f7f083d1f9d42fc))
- make ordering deterministic ([`4f4ee48`](https://github.com/jolars/eunoia/commit/4f4ee48a8fd9a6ead54dfa1c30831c8a59e62904))
- **web:** handle infinite loop in web app ([`8f36ea4`](https://github.com/jolars/eunoia/commit/8f36ea48c3b8139a3cb127fbab550ceb05f18e74))

### Performance Improvements
- add gradients for squares and rectangles too ([`50a74f3`](https://github.com/jolars/eunoia/commit/50a74f343bc12bb2d7a958f6dae482fbba0c19ca))
- wire in analytical gradient for complement case ([`402f92e`](https://github.com/jolars/eunoia/commit/402f92e852788b9f58995dd14e76261f219412a7))
## [0.11.0](https://github.com/jolars/eunoia/compare/v0.10.0...v0.11.0) (2026-05-06)

### Features
- introduce `max_sets` option to cap max sets ([`f30ed14`](https://github.com/jolars/eunoia/commit/f30ed14613eac9b1339eda60d5e9cd9bd1e8eab4))
- add NPM package ([`ad1c3e7`](https://github.com/jolars/eunoia/commit/ad1c3e71cebdd25b310e10dd5dd53bac10d0a15d))
- implement `from_shapes` and sanitize API ([`6f5dab1`](https://github.com/jolars/eunoia/commit/6f5dab1030d244e53ce351dc332da60b63a17561))

### Performance Improvements
- add analytical gradients for all smooth losses ([`f973acf`](https://github.com/jolars/eunoia/commit/f973acf5eb6ee08bfc78cac47ceddc6a13fb749d)), closes [#39](https://github.com/jolars/eunoia/issues/39)
## [0.10.0](https://github.com/jolars/eunoia/compare/v0.9.0...v0.10.0) (2026-05-04)

### Features
- **plotting:** provide stable iterator and harmonize API ([`55027cd`](https://github.com/jolars/eunoia/commit/55027cd98add36524aa1dfa3e4f7d568c6d55380))
- add inscribed-rectangle method for labeling ([`b1abeb5`](https://github.com/jolars/eunoia/commit/b1abeb5b58f300aaaefded39e37c400241fbd606))
## [0.9.0](https://github.com/jolars/eunoia/compare/v0.8.0...v0.9.0) (2026-05-04)

### Features
- clean up plotting API ([`d0a066c`](https://github.com/jolars/eunoia/commit/d0a066c21aa80804a66d9eaabcbf01e526693c3c))
- **plotting:** improve and fix plotting API ([`8fd5293`](https://github.com/jolars/eunoia/commit/8fd52930328b52882f547c6fc93dcccb9f0a3d2e))

### Bug Fixes
- **plotting:** deal with polygons contained and holes ([`b5c78c8`](https://github.com/jolars/eunoia/commit/b5c78c8dc25ddb267a2a6e4fce4a06b477e121c2))
## [0.8.0](https://github.com/jolars/eunoia/compare/v0.7.0...v0.8.0) (2026-05-01)

### Breaking changes
- make inclusive_areas lazy ([`36a4b47`](https://github.com/jolars/eunoia/commit/36a4b477190231b1f2960528d3a785ef92d5c41b))
- return geometric params for ellipses, add optim-spec one ([`36f3325`](https://github.com/jolars/eunoia/commit/36f3325bf9910f5e9c08564cd5fa0dbcececcf26))

### Features
- return geometric params for ellipses, add optim-spec one ([`36f3325`](https://github.com/jolars/eunoia/commit/36f3325bf9910f5e9c08564cd5fa0dbcececcf26))

### Performance Improvements
- make inclusive_areas lazy ([`36a4b47`](https://github.com/jolars/eunoia/commit/36a4b477190231b1f2960528d3a785ef92d5c41b))
- don't map out all 2^n - 1 areas ([`64a64d1`](https://github.com/jolars/eunoia/commit/64a64d11e2bc944c78e1d504591b34cc683ced11))
## [0.7.0](https://github.com/jolars/eunoia/compare/v0.6.0...v0.7.0) (2026-05-01)

### Features
- **wasm:** add square as a shape ([`36856e2`](https://github.com/jolars/eunoia/commit/36856e262efa242ea5454783a34773a91abfb73c))
- lower tolerance to 1e-3 ([`54c58ad`](https://github.com/jolars/eunoia/commit/54c58adf24e571fe655d02ffd8c6fc2e61a6773c))
- use venn warm start for squares too ([`ac80e07`](https://github.com/jolars/eunoia/commit/ac80e0752ad727cc8d676d1ed86530c1130be210))

### Bug Fixes
- handle lifetiime issues ([`caa29fe`](https://github.com/jolars/eunoia/commit/caa29fe07dedf42273650531b38734829c9f0a24))
## [0.6.0](https://github.com/jolars/eunoia/compare/v0.5.0...v0.6.0) (2026-04-30)

### Breaking changes
- allow generating venn diagrams with squares ([`dc0a84f`](https://github.com/jolars/eunoia/commit/dc0a84fa1e748864bcd84ec4adb7d61b4c6e2314))

### Features
- allow generating venn diagrams with squares ([`dc0a84f`](https://github.com/jolars/eunoia/commit/dc0a84fa1e748864bcd84ec4adb7d61b4c6e2314))
- support circles in euler diagrams ([`3da4604`](https://github.com/jolars/eunoia/commit/3da4604cc1db406476b05c1e8ee1b96254a7b486))
- **plotting:** add unified API for plotting details ([`4355b43`](https://github.com/jolars/eunoia/commit/4355b43a5af8c269d48f8dd4de05099847adbb09))
- add smooth loss types ([`c9c748d`](https://github.com/jolars/eunoia/commit/c9c748dfae4925d97929e40b5a69288774b2ba32))
- switch to nelder mead for non-smooth losses ([`74965d8`](https://github.com/jolars/eunoia/commit/74965d8564dd55303eba5935c7e0cea30e16c663))
- make all loss functions scale-invariant ([`d146587`](https://github.com/jolars/eunoia/commit/d146587aa662bf80e7db1a72f46cf29cf4c2b78b))

### Performance Improvements
- add analytical gradient computation to square ([`41b5f3a`](https://github.com/jolars/eunoia/commit/41b5f3a6498c60b009cce0b5ea617d4a616e8f7f))
- prune ellipse intersections too ([`edac17e`](https://github.com/jolars/eunoia/commit/edac17e845bef41764859494c20f1ffe99fdec97))
## [0.5.0](https://github.com/jolars/eunoia/compare/v0.4.0...v0.5.0) (2026-04-29)

### Breaking changes
- drop SA fallback and TrustRegion, us LM as default ([`d7ef0d2`](https://github.com/jolars/eunoia/commit/d7ef0d23508dfd243f4bc24234fcf5e72e13b181))

### Features
- support venn diagrams too ([`8da00ea`](https://github.com/jolars/eunoia/commit/8da00ea4e33d13e2c7996b18f2f52636c0738af6))
- add CMA-ES as fallback ([`9f72eb4`](https://github.com/jolars/eunoia/commit/9f72eb4bda36c4ba80092d4f010700c1e91b1659))
- drop SA fallback and TrustRegion, us LM as default ([`d7ef0d2`](https://github.com/jolars/eunoia/commit/d7ef0d23508dfd243f4bc24234fcf5e72e13b181))
- drop conjugate gradient descent solver ([`21532cc`](https://github.com/jolars/eunoia/commit/21532cc0e152432bd475ef223e09d9f35fd9a43c))

### Bug Fixes
- harden normalization of layout ([`15f1a10`](https://github.com/jolars/eunoia/commit/15f1a1086348890d6e287830b2508fbfcb103c4a))
- pre-normalize fit before packing and clustering ([`df3f0d5`](https://github.com/jolars/eunoia/commit/df3f0d504437679ec7127fce2c4ef48b0f3a87ea))
- fix quick rejection for ellipses ([`dfbad07`](https://github.com/jolars/eunoia/commit/dfbad07c3d8df0857de4be1f214d56be2f66b4cf))
- use LM in MDS initial solver too ([`8589141`](https://github.com/jolars/eunoia/commit/85891413d26d09c49490fdc03f7912be1aa5017d))

### Performance Improvements
- add deterministic venn diagram start ([`1c4f783`](https://github.com/jolars/eunoia/commit/1c4f7839ba1bac59a0a3617bea760ba839c08122))
## [0.4.0](https://github.com/jolars/eunoia/compare/v0.3.0...v0.4.0) (2026-04-28)

### Features
- add normalized SSE as loss, make it default ([`d55251e`](https://github.com/jolars/eunoia/commit/d55251e3464faf08dfd7cd6750267e31a8fff1fc))
- expose tolerance for solver ([`1803fed`](https://github.com/jolars/eunoia/commit/1803fed934837608799fb5c2f248e40dc1d12bb7))

### Bug Fixes
- avoid double-counting areas when overlapping ([`c46725d`](https://github.com/jolars/eunoia/commit/c46725d95ec61c6c64bbb3d4e95910f808748f50))
- deprecate and move away from faulty area functions ([`1fc2751`](https://github.com/jolars/eunoia/commit/1fc275133d89b01c96e051691b542a8b34efd819)), closes [#38](https://github.com/jolars/eunoia/issues/38)

### Performance Improvements
- add analytical gradients for squared-error losses ([`9e0e6b1`](https://github.com/jolars/eunoia/commit/9e0e6b1f5d837a145fea5fa60d7097eb30a43ed9))
## [0.3.0](https://github.com/jolars/eunoia/compare/v0.2.0...v0.3.0) (2026-04-27)

### Features
- make lbfgs the default optimizer ([`859ee66`](https://github.com/jolars/eunoia/commit/859ee669b7d8e519d530bdfe08f8f9bad0b71af7))

### Bug Fixes
- fix underflow in cubic solver ([`f1b0c39`](https://github.com/jolars/eunoia/commit/f1b0c39553fbedb425e918f56720f8437bb0efdd))
## [0.2.0](https://github.com/jolars/eunoia/compare/v0.1.1...v0.2.0) (2026-04-27)

### Features
- pool solvers and expose in API ([`fa1bc14`](https://github.com/jolars/eunoia/commit/fa1bc147ea7e820fb73ce2d167ccfb0bbf105a61))

### Bug Fixes
- parallelize outer loop ([`7104528`](https://github.com/jolars/eunoia/commit/71045285242470ddc05a2ff534a7349c62fc6695))
- correctly normalize ellipse rotation ([`99b8787`](https://github.com/jolars/eunoia/commit/99b8787abf38d20e1c9292857d3820e23c7554ce))
- fix logic in set clustering ([`c877319`](https://github.com/jolars/eunoia/commit/c877319c1ec050a893eb68e0284f11d7e5bd7fd9))
- fix gradient computation in initial MDS loss ([`5bd8323`](https://github.com/jolars/eunoia/commit/5bd83239e7037d707e9375bf43e46537b8c61525))
## [0.1.1](https://github.com/jolars/eunoia/compare/v0.1.0...v0.1.1) (2026-04-25)

### Bug Fixes
- parallelize initial layout ([`ee79b51`](https://github.com/jolars/eunoia/commit/ee79b510e8d407c7c3679db4c7e5a2bf450b17b5))
- remove unwanted println output ([`e4638f1`](https://github.com/jolars/eunoia/commit/e4638f1ccb7cfaebd332e03c63ba12852ea910f6))
- avoid randomness in iterating over hashmaps ([`82c090b`](https://github.com/jolars/eunoia/commit/82c090b69301a8ef50bb7e4eb46c3b58f6df1a5e))
