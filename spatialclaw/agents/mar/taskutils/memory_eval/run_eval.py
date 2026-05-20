# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from dataclasses import dataclass
import sys
from pathlib import Path
import subprocess

sys.stdout.reconfigure(line_buffering=True)
DASH_PORT = os.getenv("DASH_PORT", "8265")
SERVE_PORT = os.getenv("SERVE_PORT", "8000")
REVERSED = os.getenv("REVERSED", 0)
SERVER_BACKEND = os.getenv("EVAL_SERVER_BACKEND", "sglang")
SERVER_PYTHON = os.getenv("EVAL_SERVER_PYTHON", sys.executable)
SERVER_ARGS = os.getenv("EVAL_SERVER_ARGS", "").strip()
SAVE_SUFFIX = os.getenv("EVAL_SAVE_SUFFIX", "").strip()
FORCE = os.getenv("EVAL_FORCE", "0").strip().lower() in {"1", "true", "yes", "y"}


def safe_kill_process_group(proc, sig=2):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except ProcessLookupError:
        pass


def env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_list(name):
    value = os.getenv(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class ENV:
    # config for direct generation
    MAX_INPUT_LEN: int = 120000
    MAX_OUTPUT_LEN: int = 10000
    # Config for memory agent
    RECURRENT_MAX_CONTEXT_LEN: int = None
    RECURRENT_CHUNK_SIZE: int = None
    RECURRENT_MAX_NEW: int = None

    def setenv(self):
        if not hasattr(self, "_environ"):
            self._environ = {}
        for k, v in self.__dict__.items():
            if v is not None and k != "_environ":
                os.environ[k] = str(v)
                self._environ[k] = str(v)
                print(f"set {k}={v}")

    def unsetenv(self):
        for k in self._environ:
            os.environ[k] = self._environ[k]
        self._environ = {}

TEST_NUM_DOCS = [50, 100, 200, 400, 800, 1600, 3200, 6400]
RULER_TEST_TASKS = [
    f"eval_{ds}_{num_docs}" \
    for ds in ['hotpotqa', '2wikimultihopqa'] \
    for num_docs in TEST_NUM_DOCS
]

class Config:
    SERVE_TAG = "__serve"

    def __init__(self, name, ckpt, tp, method, env, concur=1024, tokenizer=None):
        self.name = name
        self.save_name = f"{name}{SAVE_SUFFIX}" if SAVE_SUFFIX else name
        self.ckpt = ckpt
        self.tokenizer = tokenizer or ckpt
        from pathlib import Path

        if Path(self.ckpt).is_dir():
            self.model = Path(self.ckpt).name
        else:
            self.model = self.ckpt
        self.method = method
        self.tp = tp
        self.env = env
        self.concur = concur
        self.test_process = {}

    def serve(self, wait=True):
        self.dp = env_int("EVAL_DP", int(8 / int(self.tp)))
        if SERVER_BACKEND == "vllm":
            cmd = (
                f"{SERVER_PYTHON} -m vllm.entrypoints.openai.api_server "
                f"--model {self.ckpt} "
                f"--tensor-parallel-size {self.tp} "
                f"--served-model-name {Path(self.ckpt).name} "
                f"--port {SERVE_PORT}"
            )
        elif SERVER_BACKEND == "sglang":
            cmd = (
                f"{SERVER_PYTHON} -m sglang.launch_server "
                f"--model-path {self.ckpt} "
                f"--tensor-parallel-size {self.tp} "
                f"--served-model-name {Path(self.ckpt).name} "
                f"--port {SERVE_PORT} "
                f"--data-parallel-size {self.dp}"
            )
        else:
            raise ValueError(f"Unsupported EVAL_SERVER_BACKEND={SERVER_BACKEND}")
        if SERVER_ARGS:
            cmd = f"{cmd} {SERVER_ARGS}"
        print("serving command:")
        print(cmd)
        if wait:
            os.system(f"yes | serve shutdown -a http://localhost:{DASH_PORT}")
            # setsid so that it can be interrupted
            serve_p = subprocess.Popen(cmd.split(), preexec_fn=os.setsid)
            self.test_process[self.SERVE_TAG] = serve_p
            while True:
                print("try to conntect...")
                p = subprocess.run(["curl", "-m", "100000000", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
                if p.returncode != 0:
                    print("waiting...")
                    time.sleep(5)
                elif rf'"id":"{self.model}"' not in p.stdout.decode():
                    print("model not found, maybe shutting down previous server...")
                    time.sleep(5)
                else:
                    print("connected")
                    break
        else:
            p = subprocess.run(["curl", "-m", "10", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
            if p.returncode != 0:
                print("server not started")
                exit(1)
        print(p.stdout)

    def run(self, tests, serve=True, force=False):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.env.setenv()
        self.serve(serve)
        concur = self.concur
        for test in tests:
            if test in RULER_TEST_TASKS:
                cmd = f"""python3 test_qa.py --model {self.model}\
                    --name {test} \
                    --save_dir results/{test} \
                    --save_file {self.save_name} \
                    --tokenizer {self.tokenizer} \
                    --api {self.method} \
                    --n_proc {concur}"""
            else:
                print("=" * 20 + f"Not Implemented Task {test}, please check" + "=" * 20)
                continue
            if force or FORCE:
                cmd += " --force"
            p = subprocess.Popen(cmd, shell=True)
            self.test_process[test] = p
            p.wait()
            self.test_process[test].wait()
        self.env.unsetenv()
        if serve:
            safe_kill_process_group(self.test_process[self.SERVE_TAG], 2)
            try:
                self.test_process[self.SERVE_TAG].wait(30)
            except:
                self.test_process[self.SERVE_TAG].kill()
        print("all tests finished")

    def __del__(self):
        for k, p in self.test_process.items():
            if k == self.SERVE_TAG:
                safe_kill_process_group(p, 2)
            else:
                if p.poll() is None:
                    p.kill()

L1 = Config(
    name="L1-120k+10k",
    ckpt=f"Tongyi-Zhiwen/QwenLong-L1-32B",
    tp=4,
    method="openai",
    concur=128,
    env=ENV(),
)

R1_14B = Config(
    name="R1-14B-120k+10k-openai",
    ckpt=f"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(),
)

R1_7B = Config(
    name="R1-7B-120k+10k",
    ckpt=f"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_7B_1M = Config(
    name="Qwen-7B-1M",
    ckpt=f"Qwen/Qwen2.5-7B-Instruct-1M",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(MAX_INPUT_LEN=990000, MAX_OUTPUT_LEN=10000),
)

Qwen25_14B_1M = Config(
    name="Qwen-14B-1M",
    ckpt=f"Qwen/Qwen2.5-14B-Instruct-1M",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(MAX_INPUT_LEN=990000, MAX_OUTPUT_LEN=10000),
)

Qwen3_4B_128k = Config(
    name="Qwen3-4B-128k",
    ckpt=f"Qwen/Qwen3-4B",
    tp=8,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen3_8B_128k = Config(
    name="Qwen3-8B-128k",
    ckpt=f"Qwen/Qwen3-8B",
    tp=8,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_7B_128k = Config(
    name="Qwen-7B-128k",
    ckpt="Qwen/Qwen2.5-7B-Instruct",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_3B_128k = Config(
    name="Qwen-3B-128k",
    ckpt="Qwen/Qwen2.5-3B-Instruct",
    tp=8,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_14B_128k = Config(
    name="Qwen-14B-128k",
    ckpt=f"Qwen/Qwen2.5-14B-Instruct",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_14B_MemAgent = Config(
    name="Qwen-14B-MemAgent",
    ckpt=f"Qwen/Qwen2.5-14B-Instruct",
    tp=4,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

Qwen25_3B_MemAgent = Config(
    name="Qwen-3B-MemAgent",
    ckpt="Qwen/Qwen2.5-3B-Instruct",
    tp=4,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

Qwen25_3B_ReMemR1 = Config(
    name="Qwen-3B-ReMemR1",
    ckpt="Qwen/Qwen2.5-3B-Instruct",
    tp=4,
    method="rememr1",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=2048),
)

MemAgent_7B = Config(
    name="MemAgent-7B-vanilla-em",
    ckpt=f"BytedTsinghua-SIA/RL-MemoryAgent-7B",
    tp=4,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

MemAgent_3B = Config(
    name="MemAgent-3B-vanilla-em",
    ckpt=os.getenv("MEMAGENT_3B_CKPT", "CHECKPOINT_PATH"),
    tp=env_int("MEMAGENT_3B_TP", 4),
    method="recurrent",
    concur=256,
    tokenizer=os.getenv("MEMAGENT_3B_TOKENIZER", os.getenv("QWEN25_3B_TOKENIZER", "Qwen/Qwen2.5-3B-Instruct")),
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

ReMemR1_3B = Config(
    name="ReMemR1-3B",
    ckpt=os.getenv("REMEMR1_3B_CKPT", "CHECKPOINT_PATH"),
    tp=env_int("REMEMR1_3B_TP", 4),
    method="rememr1",
    concur=256,
    tokenizer=os.getenv("REMEMR1_3B_TOKENIZER", os.getenv("QWEN25_3B_TOKENIZER", "Qwen/Qwen2.5-3B-Instruct")),
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=2048),
)

ReMemR1_7B = Config(
    name="ReMemR1-7B",
    ckpt=os.getenv("REMEMR1_7B_CKPT", "CHECKPOINT_PATH"),
    tp=env_int("REMEMR1_7B_TP", 4),
    method="rememr1",
    concur=256,
    tokenizer=os.getenv("REMEMR1_7B_TOKENIZER", os.getenv("QWEN25_7B_TOKENIZER", "Qwen/Qwen2.5-7B-Instruct")),
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=2048),
)

CONFIGS = [
    R1_7B,
    R1_14B,
    L1,
    Qwen25_7B_1M,
    Qwen25_14B_1M,
    Qwen25_7B_128k,
    Qwen25_3B_128k,
    Qwen3_4B_128k,
    Qwen3_8B_128k,
    Qwen25_3B_MemAgent,
    Qwen25_3B_ReMemR1,
    MemAgent_3B,
    MemAgent_7B,
    ReMemR1_3B,
    ReMemR1_7B,
]
if REVERSED == '1':
    CONFIGS = CONFIGS[::-1]
# Reverse the test models

def run_test_tasks():
    selected_models = set(env_list("EVAL_MODELS"))
    selected_tasks = env_list("EVAL_TASKS")
    task = selected_tasks if selected_tasks else RULER_TEST_TASKS
    configs = [
        c for c in CONFIGS
        if not selected_models or c.name in selected_models or c.model in selected_models
    ]
    if not configs:
        raise ValueError(f"No configs matched EVAL_MODELS={sorted(selected_models)}")
    for c in configs:
        c.run(task, serve=True, force=False)

if __name__ == "__main__":
    print(f"{SERVE_PORT=}, {DASH_PORT=}, {REVERSED=}")
    run_test_tasks()
