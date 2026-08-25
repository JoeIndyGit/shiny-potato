"""Fast integrity checks for the executed Session 9 submission."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "ERA_V5_Session_9_Loss_Functions_Output_Heads.ipynb"
RESULTS = ROOT / "session9_results.json"
README = ROOT / "README.md"
EXECUTION_LOG = ROOT / "EXECUTION_LOG.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
readme = README.read_text(encoding="utf-8")
require(EXECUTION_LOG.is_file(), "Execution evidence log is missing")
execution_log = EXECUTION_LOG.read_text(encoding="utf-8")
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
require(code_cells, "Notebook contains no code cells")
require(all(cell.get("execution_count") is not None for cell in code_cells),
        "At least one code cell has not been executed")

error_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
require(not error_outputs, f"Notebook contains {len(error_outputs)} execution errors")

stderr_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "stream"
    and output.get("name") == "stderr"
    and "".join(output.get("text", "")).strip()
]
require(not stderr_outputs, f"Notebook contains {len(stderr_outputs)} stderr warnings/errors")


def cell_text(notebook_cell_index: int) -> str:
    chunks: list[str] = []
    for output in notebook["cells"][notebook_cell_index].get("outputs", []):
        if "text" in output:
            value = output["text"]
            chunks.append("".join(value) if isinstance(value, list) else value)
        for value in output.get("data", {}).values():
            if isinstance(value, str) and output.get("data", {}).get("text/plain") == value:
                chunks.append(value)
            elif isinstance(value, list) and output.get("data", {}).get("text/plain") == value:
                chunks.append("".join(value))
    return "\n".join(chunks)


results = json.loads(RESULTS.read_text(encoding="utf-8"))
require(results["device"] == "cuda", "Final evidence JSON is not from the GPU run")
require(results["valid_targets"] < results["raw_targets"], "Padding mask did not reduce count")
require(abs(results["uniform_perplexity"] / results["V"] - 1.0) < 1e-6,
        "Perplexity is not at the uniform vocabulary anchor")
require(results["random_ppl_relative_delta"] < 0.05,
        "Random untrained perplexity is not sufficiently close to vocabulary size")
require(results["untied_params"] - results["tied_params"] == results["parameter_saving"],
        "Tied/untied saving is inconsistent")
require(results["parameter_saving"] == results["V"] * results["D"],
        "Weight tying did not save exactly V×D")
require(results["v5_head_params"] == 536_870_912, "V5 head count is incorrect")
require(results["loss_abs_delta"] < 1e-6, "Chunked loss mismatch exceeds tolerance")
require(results["max_gradient_delta"] < 1e-6, "Chunked gradient mismatch exceeds tolerance")
require(results["analytical_memory_ratio"] > 1, "Chunking did not reduce logits memory")
require(results["cuda_ordinary_mib"] is not None, "Final notebook lacks ordinary CUDA peak memory")
require(results["cuda_chunked_mib"] is not None, "Final notebook lacks chunked CUDA peak memory")
require(results["cuda_memory_ratio"] > 1, "Measured CUDA peak memory was not reduced")
require(results["t1_final"] < results["t1_initial"], "t+1 head did not learn")
require(results["t2_final"] < results["t2_initial"], "t+2 head did not learn")
require(abs(results["total_final"] - results["t1_final"] - results["t2_final"]) < 1e-6,
        "Part 2 total is not the exact loss sum")

# Cross-artifact provenance checks: the notebook, JSON, and README must describe one run.
run_contract_text = cell_text(2)
perplexity_text = cell_text(17)
memory_text = cell_text(22)
part2_text = cell_text(27)
boundary_text = cell_text(15)
final_text = cell_text(30)

require(results["torch_version"] in run_contract_text and "Tesla T4" in run_contract_text,
        "Run contract does not match the GPU results JSON")
require(f'{results["random_untrained_perplexity"]:.6e}' in perplexity_text,
        "Random-head perplexity drifted between notebook and JSON")
require(f'{results["cuda_ordinary_mib"]:.6f}' in memory_text,
        "Ordinary CUDA memory drifted between notebook and JSON")
require(f'{results["cuda_chunked_mib"]:.6f}' in memory_text,
        "Chunked CUDA memory drifted between notebook and JSON")
require(f'{results["t1_final"]:.6f}' in part2_text and f'{results["t2_final"]:.6f}' in part2_text,
        "Part 2 losses drifted between notebook and JSON")
require("ALL ASSIGNMENT GATES PASSED" in final_text,
        "Final executed notebook gate is missing")

for required_value in ["4.966307", "4.378274", "5.011539"]:
    require(required_value in boundary_text and required_value in readme,
            f"Packed-boundary value {required_value} is not synchronized")
for required_value in [
    f'{results["random_untrained_perplexity"]:.6f}',
    f'{results["cuda_ordinary_mib"]:.6f}',
    f'{results["cuda_chunked_mib"]:.6f}',
    f'{results["cuda_memory_ratio"]:.2f}',
    f'{results["t1_final"]:.4f}',
    f'{results["t2_final"]:.4f}',
]:
    require(required_value in readme,
            f"README is not synchronized with executed value {required_value}")

for required_value in [
    "Tesla T4",
    str(results["raw_targets"]),
    str(results["valid_targets"]),
    f'{results["random_untrained_perplexity"]:.6f}',
    f'{results["cuda_ordinary_mib"]:.6f}',
    f'{results["cuda_chunked_mib"]:.6f}',
    f'{results["t1_final"]:.6f}',
    f'{results["t2_final"]:.6f}',
    f'{results["total_final"]:.6f}',
]:
    require(required_value in execution_log,
            f"Execution log is not synchronized with executed value {required_value}")

print(f"PASS: {len(code_cells)} executed code cells, zero error or stderr outputs")
print("PASS: all Session 9 Parts 1 and 2 result invariants hold")
print("PASS: notebook, JSON, README, execution log, and charts describe one synchronized submission")
