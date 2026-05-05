module.exports = {
  run: [
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python -c \"from pathlib import Path; import subprocess; target=Path('vendor/ACE-Step-1.5'); target.parent.mkdir(parents=True, exist_ok=True); subprocess.check_call(['git','clone','--depth','1','https://github.com/ace-step/ACE-Step-1.5',str(target)]) if not target.exists() else subprocess.check_call(['git','-C',str(target),'pull','--ff-only'])\""
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python patch_ace_step_vendor.py"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          PYTHONUNBUFFERED: "1"
        },
        message: [
          "python download_models.py --all"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python -c \"import sys, platform, pathlib, numpy, soundfile, torch, gradio, transformers, diffusers, peft, lightning, lycoris, tensorboard, toml, modelscope, typer, torchao; intel_mac = sys.platform == 'darwin' and platform.machine() == 'x86_64'; assert (not intel_mac) or numpy.__version__.split('.')[0] == '1', f'Intel Mac requires NumPy 1.x, got {numpy.__version__}'; assert (not intel_mac) or diffusers.__version__ == '0.31.0', f'Intel Mac requires diffusers 0.31.0, got {diffusers.__version__}'; assert pathlib.Path('vendor/ACE-Step-1.5/train.py').is_file(), 'Official ACE-Step trainer missing'; print('Core and trainer Python deps ready')\""
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          PYTHONUNBUFFERED: "1"
        },
        message: [
          "python install_mflux.py"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          PYTHONUNBUFFERED: "1"
        },
        message: [
          "python install_mlx_video.py"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app/web",
        message: [
          "npm install --no-audit --no-fund",
          "npm run build"
        ]
      }
    }
  ]
}
