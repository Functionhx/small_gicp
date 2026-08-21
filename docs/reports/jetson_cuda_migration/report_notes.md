# Jetson CUDA migration report notes

## Reporting job

- Question: explain how the project moved from the current `cuda-x86` implementation to a Jetson-native `cuda-jetson` implementation, including engineering decisions, rejected experiments, validation, and measured outcomes.
- Audience: technical readers learning CUDA migration and embedded-GPU performance engineering.
- Scope: code and measurements completed on 2026-08-21, ending at commit `6e836f60c04e2cf40069b25384b15764ac7757e2`.
- Baselines: current local `cuda-x86` worktree for Jetson-only code differences; published `origin/cuda-x86` for the full GitHub commit history.
- Success criteria: the reader can reproduce the build, understand each optimization's data-flow effect, distinguish quality12 from balanced8, and audit every headline number.

## Technical-report structure mapping

| Required role | LaTeX section |
|---|---|
| Title | Title page |
| Technical summary | 技术摘要 |
| Key findings with visual evidence | 迁移流程图、最终性能分组条形图、结果表 |
| Scope, data, metric definitions | 验证设计与指标口径 |
| Methodology | 迁移阶段、Nsight 诊断、内存与内核优化 |
| Limitations and robustness | 失败实验、局限性与数值边界 |
| Recommended next steps | 后续工作 |
| Further questions | 10.3 建议的后续工作 |

## Visual contracts

### Migration flow

- Question: what sequence turned the x86 CUDA implementation into a Jetson-native pipeline?
- Takeaway: device bring-up and profiling preceded optimization; validation gates followed every stage.
- Form: TikZ process flow, six stages in a two-row snake.
- Encoding: position and arrows carry order; restrained blue/neutral styling is decorative only.

### Final latency comparison

- Question: under the unchanged live workload, how do CPU12, CPU8, CUDA quality12, and CUDA balanced8 compare by algorithm?
- Takeaway: quality12 beats CPU8 for every algorithm; balanced8 adds a documented preprocessing tradeoff and larger speedup.
- Form: grouped horizontal bar chart, zero-based latency axis, five algorithms and four configurations.
- Data: ten-run median p50 in milliseconds per frame from `evidence/optimized_full_matrix_summary.json`.
- Palette: two neutral CPU tones, blue quality baseline, orange balanced profile; series are also distinguished by grouped position and legend labels.
- QA: exact values remain in an adjacent table; the caption explicitly states that balanced8 is approximate and that ICP is unaffected by shell cap.

## Evidence inventory

- `evidence/optimized_full_matrix_summary.json`: ten-block optimized CPU12/CPU8/quality12/balanced8/FastVGICPCuda matrix.
- `evidence/base_vs_optimized_summary.json`: ten-block saved-base/stage2/quality12/balanced8 A/B.
- `evidence/report_validation.json`: report-source completeness and independent calculation checks.
- `JETSON.md` and `BENCHMARK_GPU.md`: build commands, device environment, Nsight summaries, numerical envelope, and reproduction notes.
- Git commit `6e836f6`: source-of-record implementation diff.

## Required caveats

- Performance is conditional on the observed live ROS/inference workload, not isolated peak throughput.
- `balanced8` is an explicit approximate covariance-search profile and lacks KITTI ground-truth evaluation.
- FastVGICPCuda default and project VGICP scan-to-scan use different preprocessing implementations.
- The current LM accept/reject controller and 6x6 solve remain on CPU; eliminating them requires a persistent cooperative optimizer rather than mechanical graph capture.
