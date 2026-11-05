"""build_train_command (utils/launch.py): the command /train starts, single GPU and multi-GPU.

With the installer's default Accelerate config (distributed_type NO, gpu_ids '0') 'accelerate launch
--num_processes=2' still starts one process on GPU 0; more than one GPU needs --multi_gpu and the GPU ids on the
command line. A config the user made with 'accelerate config' (MULTI_GPU on chosen GPUs, DeepSpeed, FSDP) keeps
deciding: the backend passes only --num_processes, plus GPU ids the request names.

Run from the backend folder with any Python 3.10+ (the config file tests need PyYAML):

    python -m unittest tests.test_launch
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.launch import (  # noqa: E402
    build_train_command,
    default_accelerate_config_path,
    read_accelerate_config,
)

PY, SCRIPT, CONFIG, DATASET = "python", "/b/sd_scripts/anima_train_network.py", "/b/config.toml", "/b/dataset.toml"
TAIL = [SCRIPT, f"--config_file={CONFIG}", f"--dataset_config={DATASET}"]
LAUNCH = [PY, "-m", "accelerate.commands.launch"]
INSTALLER_DEFAULT = {"compute_environment": "LOCAL_MACHINE", "distributed_type": "NO", "gpu_ids": "0",
                     "num_processes": 1}
USER_MULTI_GPU = {"compute_environment": "LOCAL_MACHINE", "distributed_type": "MULTI_GPU", "gpu_ids": "1,2",
                  "num_processes": 2}
USER_DEEPSPEED = {"compute_environment": "LOCAL_MACHINE", "distributed_type": "DEEPSPEED", "gpu_ids": "all",
                  "deepspeed_config": {"zero_stage": 2, "offload_optimizer_device": "cpu"}, "num_processes": 2}
HAS_YAML = importlib.util.find_spec("yaml") is not None


def build(**kw):
    kw.setdefault("accelerate_config", INSTALLER_DEFAULT)
    return build_train_command(PY, SCRIPT, CONFIG, DATASET, **kw)


class BuildTrainCommandTest(unittest.TestCase):
    def test_plain_python_without_accelerate(self):
        self.assertEqual(build(), [PY, *TAIL])
        self.assertEqual(build(accelerate=False, num_processes=4, gpu_ids="x"), [PY, *TAIL])

    def test_one_process_with_accelerate_is_unchanged(self):
        for config in (INSTALLER_DEFAULT, None, USER_MULTI_GPU, USER_DEEPSPEED):
            with self.subTest(config=config):
                self.assertEqual(
                    build(accelerate=True, num_processes=1, main_process_port=29501, environ={},
                          accelerate_config=config),
                    [*LAUNCH, "--num_processes=1", "--main_process_port=29501", *TAIL],
                )

    def test_two_gpus_pass_multi_gpu_and_ids(self):
        for config in (INSTALLER_DEFAULT, None):  # single-process config, or none at all
            with self.subTest(config=config):
                self.assertEqual(
                    build(accelerate=True, num_processes=2, main_process_port=29500, environ={},
                          accelerate_config=config),
                    [*LAUNCH, "--multi_gpu", "--num_processes=2", "--gpu_ids=0,1", "--main_process_port=29500",
                     *TAIL],
                )

    def test_unquoted_no_is_single_process(self):
        cmd = build(accelerate=True, num_processes=2, environ={}, accelerate_config={"distributed_type": False})
        self.assertIn("--multi_gpu", cmd)

    def test_gpu_ids_follow_cuda_visible_devices(self):
        cmd = build(accelerate=True, num_processes=2, environ={"CUDA_VISIBLE_DEVICES": "2, 3,1"})
        self.assertIn("--gpu_ids=2,3", cmd)

    def test_requested_gpu_ids(self):
        cmd = build(accelerate=True, num_processes=3, gpu_ids="1, 2,5", environ={"CUDA_VISIBLE_DEVICES": "0"})
        self.assertIn("--gpu_ids=1,2,5", cmd)
        self.assertIn("--multi_gpu", cmd)

    def test_user_multi_process_config_decides(self):
        """The reviewer's cases: a MULTI_GPU config on GPUs 1,2 and a DeepSpeed config keep the old argv (no
        --multi_gpu, which would turn DeepSpeed into DDP; no --gpu_ids, which would move the run to GPU 0)."""
        for config in (USER_MULTI_GPU, USER_DEEPSPEED, {**USER_MULTI_GPU, "gpu_ids": "all"},
                       {"distributed_type": "FSDP"}, {"distributed_type": "MEGATRON_LM"}):
            with self.subTest(config=config):
                self.assertEqual(
                    build(accelerate=True, num_processes=2, main_process_port=29500, environ={},
                          accelerate_config=config),
                    [*LAUNCH, "--num_processes=2", "--main_process_port=29500", *TAIL],
                )

    def test_requested_ids_with_a_user_config(self):
        for config in (USER_MULTI_GPU, USER_DEEPSPEED):
            with self.subTest(config=config):
                cmd = build(accelerate=True, num_processes=2, gpu_ids="0,3", environ={}, accelerate_config=config)
                self.assertEqual(cmd[3:5], ["--num_processes=2", "--gpu_ids=0,3"])
                self.assertNotIn("--multi_gpu", cmd)
        # a DeepSpeed / FSDP run on one chosen GPU
        cmd = build(accelerate=True, num_processes=1, gpu_ids="1", environ={}, accelerate_config=USER_DEEPSPEED)
        self.assertEqual(cmd[3:5], ["--num_processes=1", "--gpu_ids=1"])

    def test_unreadable_config_is_left_to_accelerate(self):
        cmd = build(accelerate=True, num_processes=2, environ={}, accelerate_config={})
        self.assertEqual(cmd[3:], ["--num_processes=2", "--main_process_port=29500", *TAIL])

    def test_one_process_on_a_requested_gpu(self):
        """accelerate_gpu_ids with one process is passed (Accelerate's simple launcher honours it), not dropped."""
        for config in (INSTALLER_DEFAULT, None):
            with self.subTest(config=config):
                self.assertEqual(
                    build(accelerate=True, num_processes=1, gpu_ids=" 1 ", environ={"CUDA_VISIBLE_DEVICES": "0"},
                          accelerate_config=config),
                    [*LAUNCH, "--num_processes=1", "--gpu_ids=1", "--main_process_port=29500", *TAIL],
                )
        self.assertNotIn("--gpu_ids=", " ".join(build(accelerate=True, num_processes=1, gpu_ids="", environ={})))

    def test_invalid_requests(self):
        cases = [
            {"num_processes": 0},
            {"num_processes": 2, "gpu_ids": "0"},  # fewer ids than processes
            {"num_processes": 2, "gpu_ids": "0,0"},  # duplicate
            {"num_processes": 2, "gpu_ids": "0,x"},
            {"num_processes": 2, "environ": {"CUDA_VISIBLE_DEVICES": "3"}},  # only one visible GPU
            {"num_processes": 1, "gpu_ids": "x"},
            {"num_processes": 1, "gpu_ids": "0,1"},  # more ids than processes
            {"num_processes": 1, "gpu_ids": "x", "accelerate_config": USER_DEEPSPEED},
            # a MULTI_GPU config: accelerate refuses one GPU id ("Less than two GPU ids")
            {"num_processes": 1, "gpu_ids": "1", "accelerate_config": USER_MULTI_GPU},
            # a MULTI_GPU config listing fewer GPUs than processes would start a process without a GPU
            {"num_processes": 3, "accelerate_config": USER_MULTI_GPU},
            {"num_processes": 2, "accelerate_config": {**USER_MULTI_GPU, "gpu_ids": 0}},
        ]
        for kw in cases:
            kw = {"environ": {}, **kw}
            with self.subTest(**kw), self.assertRaises(ValueError):
                build(accelerate=True, **kw)


class DefaultConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_path_follows_hf_home(self):
        self.assertEqual(default_accelerate_config_path({"HF_HOME": str(self.root)}),
                         self.root / "accelerate" / "default_config.yaml")
        self.assertEqual(default_accelerate_config_path({"XDG_CACHE_HOME": str(self.root)}),
                         self.root / "huggingface" / "accelerate" / "default_config.yaml")
        self.assertEqual(default_accelerate_config_path({}),
                         Path("~/.cache/huggingface/accelerate/default_config.yaml").expanduser())

    def test_missing_config(self):
        self.assertIsNone(read_accelerate_config(self.root / "default_config.yaml"))
        env = {"HF_HOME": str(self.root)}
        cmd = build_train_command(PY, SCRIPT, CONFIG, DATASET, accelerate=True, num_processes=2, environ=env)
        self.assertIn("--multi_gpu", cmd)

    @unittest.skipUnless(HAS_YAML, "PyYAML is not installed in this Python")
    def test_reads_the_default_config_of_the_environment(self):
        path = self.root / "accelerate" / "default_config.yaml"
        path.parent.mkdir()
        path.write_text("compute_environment: LOCAL_MACHINE\ndistributed_type: MULTI_GPU\ngpu_ids: 1,2\n"
                        "num_processes: 2\n", encoding="utf-8")
        self.assertEqual(read_accelerate_config(path)["gpu_ids"], "1,2")
        env = {"HF_HOME": str(self.root)}
        cmd = build_train_command(PY, SCRIPT, CONFIG, DATASET, accelerate=True, num_processes=2, environ=env)
        self.assertEqual(cmd[3:], ["--num_processes=2", "--main_process_port=29500", *TAIL])

        path.write_text("distributed_type: 'NO'\ngpu_ids: '0'\n", encoding="utf-8")
        cmd = build_train_command(PY, SCRIPT, CONFIG, DATASET, accelerate=True, num_processes=2, environ=env)
        self.assertEqual(cmd[3:6], ["--multi_gpu", "--num_processes=2", "--gpu_ids=0,1"])

        path.write_text("distributed_type: [unclosed\n", encoding="utf-8")
        self.assertEqual(read_accelerate_config(path), {})


if __name__ == "__main__":
    unittest.main()
