"""
UniRig auto-rigging pipeline wrapper.

Calls UniRig's three shell scripts as subprocesses using an isolated
Python 3.11 virtualenv (separate from the main TRELLIS.2 venv).

Workflow:
    1. generate_skeleton.sh  -> skeleton.fbx
    2. generate_skin.sh      -> skin.fbx
    3. merge.sh              -> rigged.glb (original geometry + skeleton + skin weights)
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class RiggingPipeline:
    """Wrap UniRig CLI scripts for automatic mesh rigging."""

    def __init__(self, unirig_home=None, unirig_python=None):
        self.unirig_home = Path(
            unirig_home or os.environ.get("UNIRIG_HOME", "/opt/UniRig")
        )
        self.unirig_python = (
            unirig_python
            or os.environ.get("UNIRIG_PYTHON", "/opt/unirig_venv/bin/python")
        )

    def rig_mesh(
        self,
        glb_path: str,
        output_dir: str,
        seed: int = 12345,
    ) -> str:
        """Run the full rigging pipeline: skeleton -> skin -> merge.

        Args:
            glb_path: Path to the input GLB file (with PBR textures).
            output_dir: Directory for intermediate and final output files.
            seed: Random seed for skeleton generation reproducibility.

        Returns:
            Path to the rigged GLB file.
        """
        glb_path = str(glb_path)
        output_dir = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        skeleton_fbx = os.path.join(output_dir, "skeleton.fbx")
        skin_fbx = os.path.join(output_dir, "skin.fbx")
        rigged_glb = os.path.join(output_dir, "rigged.glb")
        npz_dir = os.path.join(output_dir, "npz")
        os.makedirs(npz_dir, exist_ok=True)

        # Step 1: Generate skeleton
        logger.info("UniRig step 1/3: generating skeleton...")
        self._run_script(
            "launch/inference/generate_skeleton.sh",
            input=glb_path,
            output=skeleton_fbx,
            seed=str(seed),
            npz_dir=npz_dir,
        )

        # Step 2: Generate skin weights
        logger.info("UniRig step 2/3: generating skin weights...")
        self._run_script(
            "launch/inference/generate_skin.sh",
            input=skeleton_fbx,
            output=skin_fbx,
            npz_dir=npz_dir,
        )

        # Step 3: Merge onto original mesh
        logger.info("UniRig step 3/3: merging rig onto original mesh...")
        self._run_script(
            "launch/inference/merge.sh",
            source=skin_fbx,
            target=glb_path,
            output=rigged_glb,
        )

        if not os.path.isfile(rigged_glb):
            raise RuntimeError(
                f"UniRig merge did not produce output file: {rigged_glb}"
            )

        logger.info("UniRig rigging complete: %s", rigged_glb)
        return rigged_glb

    def _run_script(self, script_name: str, **kwargs) -> subprocess.CompletedProcess:
        """Execute a UniRig shell script with the unirig venv on PATH."""
        venv_bin = str(Path(self.unirig_python).parent)
        env = os.environ.copy()
        env["PATH"] = venv_bin + ":" + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(Path(self.unirig_python).parent.parent)
        # Ensure HF_HOME is forwarded so model weights use the shared cache
        env.setdefault("HF_HOME", "/workspace/cache/huggingface")

        cmd = ["bash", str(self.unirig_home / script_name)]
        for k, v in kwargs.items():
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])

        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(self.unirig_home),
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error("UniRig script %s failed:\nstdout: %s\nstderr: %s",
                         script_name, result.stdout, result.stderr)
            raise RuntimeError(
                f"UniRig {script_name} failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:] if result.stderr else result.stdout[-2000:]}"
            )

        return result
