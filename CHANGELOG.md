# Changelog

## [0.5.0](https://github.com/jolars/eunoia/compare/v0.4.0...v0.5.0) (2026-04-29)

### Breaking changes
- drop SA fallback and TrustRegion, us LM as default ([`d7ef0d2`](https://github.com/jolars/eunoia/commit/d7ef0d23508dfd243f4bc24234fcf5e72e13b181))

### Features
- support venn diagrams too ([`8da00ea`](https://github.com/jolars/eunoia/commit/8da00ea4e33d13e2c7996b18f2f52636c0738af6))
- add CMA-ES as fallback ([`9f72eb4`](https://github.com/jolars/eunoia/commit/9f72eb4bda36c4ba80092d4f010700c1e91b1659))
- drop SA fallback and TrustRegion, us LM as default ([`d7ef0d2`](https://github.com/jolars/eunoia/commit/d7ef0d23508dfd243f4bc24234fcf5e72e13b181))
- drop conjugate gradient descent solver ([`21532cc`](https://github.com/jolars/eunoia/commit/21532cc0e152432bd475ef223e09d9f35fd9a43c))

### Bug Fixes
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
