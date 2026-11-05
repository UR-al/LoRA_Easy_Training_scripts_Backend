"""The installer's default Accelerate config (installer.setup_accelerate) loads with Accelerate's own parser.

An older installer wrote 'downcase_fp16', which Accelerate 1.x rejects ("unknown keys"), so every 'accelerate launch'
on a new install stopped before training. Since the installer skips an existing config, a re-install never fixed it:
it now removes that key from an existing config and leaves every other config untouched.

Run from the backend folder with the training venv (the parser tests need accelerate; they are skipped without it):

    sd_scripts\\venv\\Scripts\\python -m unittest tests.test_installer_accelerate
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import installer  # noqa: E402
from utils.launch import (  # noqa: E402
    build_train_command,
    default_accelerate_config_path,
    read_accelerate_config,
)

HAS_ACCELERATE = importlib.util.find_spec("accelerate") is not None

OLD_INSTALLER_CONFIG = (
    "command_file: null\n"
    "commands: null\n"
    "compute_environment: LOCAL_MACHINE\n"
    "deepspeed_config: {}\n"
    "distributed_type: 'NO'\n"
    "downcase_fp16: 'NO'\n"
    "dynamo_backend: 'NO'\n"
    "fsdp_config: {}\n"
    "gpu_ids: '0'\n"
    "machine_rank: 0\n"
    "main_process_ip: null\n"
    "main_process_port: null\n"
    "main_training_function: main\n"
    "megatron_lm_config: {}\n"
    "mixed_precision: bf16\n"
    "num_machines: 1\n"
    "num_processes: 1\n"
    "rdzv_backend: static\n"
    "same_network: true\n"
    "tpu_name: null\n"
    "tpu_zone: null\n"
    "use_cpu: false"
)


def load_with_accelerate(path: Path):
    from accelerate.commands.config.config_args import load_config_from_file

    return load_config_from_file(str(path))


def validated_launch_args(config: Path, *launch_args: str):
    """accelerate launch's own argument parsing and validation, without starting anything."""
    from accelerate.commands.launch import _validate_launch_command, launch_command_parser

    args = launch_command_parser().parse_args(["--config_file", str(config), *launch_args, "train_network.py"])
    _validate_launch_command(args)
    return args


def backend_launch_args(config: Path, num_processes: int, gpu_ids=None) -> list:
    """The accelerate options of the command /train builds (utils/launch.py) when config is the default Accelerate
    config, without python -m and the script."""
    cmd = build_train_command("python", "train_network.py", "c.toml", "d.toml", accelerate=True,
                              num_processes=num_processes, gpu_ids=gpu_ids, environ={},
                              accelerate_config=read_accelerate_config(config))
    return cmd[3:cmd.index("train_network.py")]


USER_MULTI_GPU_CONFIG = """compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
gpu_ids: 1,2
mixed_precision: bf16
num_machines: 1
num_processes: 2
machine_rank: 0
main_training_function: main
use_cpu: false
"""

USER_DEEPSPEED_CONFIG = """compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  offload_optimizer_device: cpu
  zero_stage: 2
distributed_type: DEEPSPEED
gpu_ids: all
mixed_precision: bf16
num_machines: 1
num_processes: 2
machine_rank: 0
main_training_function: main
use_cpu: false
"""


class AccelerateConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Path(self._tmp.name, "huggingface", "accelerate", "default_config.yaml")

    def tearDown(self):
        self._tmp.cleanup()

    def test_new_config_has_no_unknown_or_tpu_keys(self):
        installer.setup_accelerate("linux", path=self.config)
        text = self.config.read_text(encoding="utf-8")
        keys = {line.split(":", 1)[0] for line in text.splitlines() if line.strip()}
        self.assertNotIn("downcase_fp16", keys)
        self.assertFalse({"tpu_name", "tpu_zone", "command_file", "commands"} & keys, keys)
        self.assertIn("mixed_precision", keys)

    @unittest.skipUnless(HAS_ACCELERATE, "accelerate is not installed in this Python")
    def test_new_config_loads_with_accelerate(self):
        installer.setup_accelerate("windows", path=self.config)
        loaded = load_with_accelerate(self.config)
        self.assertEqual(loaded.distributed_type.value, "NO")
        self.assertEqual(loaded.mixed_precision, "bf16")
        self.assertEqual(loaded.num_processes, 1)
        self.assertEqual(loaded.num_machines, 1)
        self.assertFalse(loaded.use_cpu)

    @unittest.skipUnless(HAS_ACCELERATE, "accelerate is not installed in this Python")
    def test_old_installer_config_is_rejected_by_accelerate(self):
        """The reviewer's finding, so the repair below is known to matter."""
        self.config.parent.mkdir(parents=True)
        self.config.write_text(OLD_INSTALLER_CONFIG, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "downcase_fp16"):
            load_with_accelerate(self.config)

    def test_existing_config_with_the_old_typo_is_repaired(self):
        self.config.parent.mkdir(parents=True)
        self.config.write_text(OLD_INSTALLER_CONFIG, encoding="utf-8")
        installer.setup_accelerate("linux", path=self.config)
        text = self.config.read_text(encoding="utf-8")
        self.assertNotIn("downcase_fp16", text)
        # Everything else stays as the user had it
        self.assertEqual(text.splitlines(), [line for line in OLD_INSTALLER_CONFIG.splitlines()
                                             if not line.startswith("downcase_fp16")])
        if HAS_ACCELERATE:
            self.assertEqual(load_with_accelerate(self.config).mixed_precision, "bf16")

    def test_existing_valid_config_is_left_alone(self):
        self.config.parent.mkdir(parents=True)
        mine = "compute_environment: LOCAL_MACHINE\r\ndistributed_type: MULTI_GPU\r\nnum_processes: 2\r\n"
        self.config.write_bytes(mine.encode("utf-8"))
        installer.setup_accelerate("linux", path=self.config)
        self.assertEqual(self.config.read_bytes(), mine.encode("utf-8"))

    @unittest.skipUnless(HAS_ACCELERATE, "accelerate is not installed in this Python")
    def test_launch_with_the_default_config(self):
        """With the default config, --num_processes alone still runs one process; --multi_gpu and --gpu_ids (what
        the backend now passes for more than one GPU) give a multi-GPU launch."""
        installer.setup_accelerate("linux", path=self.config)

        single = validated_launch_args(self.config, "--num_processes=2")
        self.assertFalse(single.multi_gpu)

        multi = validated_launch_args(self.config, "--multi_gpu", "--num_processes=2", "--gpu_ids=0,1")
        self.assertTrue(multi.multi_gpu)
        self.assertEqual(multi.num_processes, 2)
        self.assertEqual(multi.gpu_ids, "0,1")

        with self.assertRaisesRegex(ValueError, "Less than two GPU ids"):
            validated_launch_args(self.config, "--multi_gpu", "--num_processes=2")  # gpu_ids '0' from the config

    @unittest.skipUnless(HAS_ACCELERATE, "accelerate is not installed in this Python")
    def test_backend_command_starts_one_child_per_gpu(self):
        from accelerate.utils.launch import prepare_multi_gpu_env, prepare_simple_launcher_cmd_env

        installer.setup_accelerate("linux", path=self.config)

        two = validated_launch_args(self.config, *backend_launch_args(self.config, 2))
        self.assertTrue(two.multi_gpu)
        env = prepare_multi_gpu_env(two)  # torchrun settings: nproc_per_node children, the chosen GPUs
        self.assertEqual(two.nproc_per_node, "2")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")

        one = validated_launch_args(self.config, *backend_launch_args(self.config, 1))
        self.assertFalse(one.multi_gpu)
        _, env = prepare_simple_launcher_cmd_env(one)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

        # one process on the GPU the request names: the simple launcher honours --gpu_ids
        one = validated_launch_args(self.config, *backend_launch_args(self.config, 1, gpu_ids="1"))
        self.assertFalse(one.multi_gpu)
        _, env = prepare_simple_launcher_cmd_env(one)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def write_config(self, text: str) -> Path:
        self.config.parent.mkdir(parents=True)
        self.config.write_text(text, encoding="utf-8")
        return self.config

    @unittest.skipUnless(HAS_ACCELERATE, "accelerate is not installed in this Python")
    def test_user_multi_gpu_config_keeps_its_gpus(self):
        """A config from 'accelerate config' that leaves GPU 0 out: the backend's launch stays on GPUs 1,2."""
        from accelerate.utils.launch import prepare_multi_gpu_env

        config = self.write_config(USER_MULTI_GPU_CONFIG)
        args = validated_launch_args(config, *backend_launch_args(config, 2))
        self.assertTrue(args.multi_gpu)
        self.assertEqual(prepare_multi_gpu_env(args)["CUDA_VISIBLE_DEVICES"], "1,2")

        # GPU ids the request names win
        args = validated_launch_args(config, *backend_launch_args(config, 2, gpu_ids="0,3"))
        self.assertTrue(args.multi_gpu)
        self.assertEqual(prepare_multi_gpu_env(args)["CUDA_VISIBLE_DEVICES"], "0,3")

    @unittest.skipUnless(HAS_ACCELERATE, "accelerate is not installed in this Python")
    def test_user_deepspeed_config_stays_deepspeed(self):
        config = self.write_config(USER_DEEPSPEED_CONFIG)
        for gpu_ids in (None, "0,1"):
            with self.subTest(gpu_ids=gpu_ids):
                args = validated_launch_args(config, *backend_launch_args(config, 2, gpu_ids=gpu_ids))
                self.assertTrue(args.use_deepspeed)
                self.assertFalse(args.multi_gpu)
                self.assertEqual(args.zero_stage, 2)
                self.assertEqual(args.offload_optimizer_device, "cpu")
                self.assertEqual(args.gpu_ids, gpu_ids or "all")

    @unittest.skipUnless(HAS_ACCELERATE, "accelerate is not installed in this Python")
    def test_default_config_path_is_accelerates(self):
        from accelerate.commands.config.config_args import default_config_file

        self.assertEqual(default_accelerate_config_path(), Path(default_config_file))


if __name__ == "__main__":
    unittest.main()
