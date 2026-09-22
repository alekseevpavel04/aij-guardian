"""Memory of this machine: whether the local model still fits the GPU (RTX 3070, 8 GB) and the RAM.

On Windows a CUDA allocation that does not fit the card is not refused: the driver moves it to
"shared GPU memory" in system RAM behind PCIe, and the program keeps running slower without failing.
"Shared" is not by itself that spill: llama.cpp pins host buffers for the experts computed on the
CPU, and Windows counts them as shared GPU memory too (0.4 GB at ubatch 2048, 0.7-0.9 GB at 4096;
they vanish with GGML_CUDA_NO_PINNED=1 while the card's own usage stays the same). A spill is
shared memory while the card is full, so the warning needs both.

Low free RAM is not by itself trouble either: most of llama-server's working set is pages of the
mapped model file, which the system drops without writing anything. Trouble is paging, pages read
back from disk every second (13.5 thousand a second once the server's RAM prompt cache filled it).
Windows counts pages taken back from its compressed memory as page input too, in bursts of several
thousand with no disk reads at all, so paging is trouble only while the disks are being read.
"""

import json
import subprocess
from dataclasses import dataclass

# free card memory below this, with shared memory in use, means allocations were pushed off the card
GPU_FREE_LIMIT_GB = 0.3
RAM_FREE_LIMIT_GB = 0.5
PAGES_IN_LIMIT = 2000  # page input per second
DISK_READ_LIMIT_MB = 20  # disk reads per second that make that page input real paging


@dataclass
class Memory:
    gpu_used_gb: float
    gpu_total_gb: float
    dedicated_gb: dict[int, float]  # pid -> GB on the card
    shared_gb: dict[int, float]  # pid -> GB in shared GPU memory
    ram_free_gb: float
    ram_total_gb: float
    pages_in_per_s: float
    disk_read_mb_per_s: float

    def warnings(self) -> list[str]:
        out = []
        if self.gpu_total_gb - self.gpu_used_gb < GPU_FREE_LIMIT_GB and self.shared_gb:
            out.append(
                f"GPU full ({self.gpu_used_gb:.1f} of {self.gpu_total_gb:.1f} GB) and "
                f"{max(self.shared_gb.values()):.2f} GB in shared memory: allocations spill to RAM"
            )
        if self.ram_free_gb < RAM_FREE_LIMIT_GB:
            out.append(f"only {self.ram_free_gb:.1f} GB of RAM free")
        if self.pages_in_per_s > PAGES_IN_LIMIT and self.disk_read_mb_per_s > DISK_READ_LIMIT_MB:
            out.append(
                f"paging: {self.pages_in_per_s:.0f} pages in, "
                f"{self.disk_read_mb_per_s:.0f} MB read from disk per second"
            )
        return out


def _powershell(command: str) -> str:
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True
    ).stdout


def measure() -> Memory:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split(",")
    counters = json.loads(
        _powershell(
            "$s = (Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage','\\GPU Process Memory(*)\\Shared Usage')"
            ".CounterSamples | Select-Object Path,CookedValue; $s | ConvertTo-Json -Compress"
        )
    )
    dedicated: dict[int, float] = {}
    shared: dict[int, float] = {}
    for sample in counters:
        path = sample["Path"].lower()
        pid = int(path.split("(pid_")[1].split("_")[0])
        target = shared if path.endswith("shared usage") else dedicated
        target[pid] = target.get(pid, 0.0) + sample["CookedValue"] / 2**30
    ram = json.loads(
        _powershell(
            "Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory,TotalVisibleMemorySize"
            " | ConvertTo-Json -Compress"
        )
    )
    # the CIM classes, not Get-Counter: counter paths like \Memory are localised on a Russian Windows;
    # one snapshot jumps by two orders of magnitude, so the median of three seconds
    samples = json.loads(
        _powershell(
            "$v = 1..3 | ForEach-Object { @{pages = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory)"
            ".PagesInputPersec; disk = (Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk"
            " -Filter \"Name='_Total'\").DiskReadBytesPersec}; Start-Sleep 1 }; $v | ConvertTo-Json -Compress"
        )
    )
    pages_in = sorted(float(x["pages"]) for x in samples)[1]
    disk_read = sorted(float(x["disk"]) for x in samples)[1] / 2**20
    return Memory(
        gpu_used_gb=float(smi[0]) / 1024,
        gpu_total_gb=float(smi[1]) / 1024,
        dedicated_gb={p: g for p, g in dedicated.items() if g > 0.05},
        shared_gb={p: g for p, g in shared.items() if g > 0.05},
        ram_free_gb=ram["FreePhysicalMemory"] / 2**20,
        ram_total_gb=ram["TotalVisibleMemorySize"] / 2**20,
        pages_in_per_s=pages_in,
        disk_read_mb_per_s=disk_read,
    )
