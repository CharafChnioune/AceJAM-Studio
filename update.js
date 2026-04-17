module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: [
        "git pull"
      ]
    }
  }, {
    when: "{{exists('app/requirements.txt')}}",
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env",
        path: "app"
      }
    }
  }, {
    when: "{{exists('app/requirements.txt')}}",
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "uv pip install -r requirements.txt"
      ]
    }
  }, {
    when: "{{exists('app/requirements.txt') && platform === 'darwin' && arch === 'arm64'}}",
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "CMAKE_ARGS=\"-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DGGML_METAL=on\" uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python==0.3.20",
        "python -c \"import llama_cpp; print(llama_cpp.__version__)\""
      ]
    }
  }, {
    when: "{{exists('app/requirements.txt') && platform === 'win32' && arch === 'x64'}}",
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "uv pip install --upgrade --force-reinstall \"https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.19/llama_cpp_python-0.3.19-cp310-cp310-win_amd64.whl\"",
        "python -c \"import llama_cpp; print(llama_cpp.__version__)\""
      ]
    }
  }, {
    when: "{{exists('app/requirements.txt') && platform === 'darwin' && arch !== 'arm64'}}",
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "CMAKE_ARGS=\"-DGGML_METAL=OFF\" uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python==0.3.20 numpy==1.26.4",
        "python -c \"import llama_cpp; print(llama_cpp.__version__)\""
      ]
    }
  }, {
    when: "{{exists('app/requirements.txt') && platform !== 'darwin' && platform !== 'win32'}}",
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "uv pip install --index-strategy unsafe-best-match --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --upgrade --force-reinstall --no-cache-dir llama-cpp-python==0.3.20",
        "python -c \"import llama_cpp; print(llama_cpp.__version__)\""
      ]
    }
  }, {
    when: "{{exists('app/requirements.txt')}}",
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "python -c \"import sys, platform, numpy, soundfile, torch, gradio, transformers, diffusers; intel_mac = sys.platform == 'darwin' and platform.machine() == 'x86_64'; assert (not intel_mac) or numpy.__version__.split('.')[0] == '1', f'Intel Mac requires NumPy 1.x, got {numpy.__version__}'; assert (not intel_mac) or diffusers.__version__ == '0.31.0', f'Intel Mac requires diffusers 0.31.0, got {diffusers.__version__}'; print('Core Python deps ready')\""
      ]
    }
  }]
}
