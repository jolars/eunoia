# Changelog

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
