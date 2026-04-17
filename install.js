module.exports = {
  run: [
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
      when: "{{platform === 'darwin' && arch === 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "CMAKE_ARGS=\"-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DGGML_METAL=on\" uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python==0.3.20",
          "python -c \"import llama_cpp; print(llama_cpp.__version__)\""
        ]
      }
    },
    {
      when: "{{platform !== 'darwin' || arch !== 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install --index-strategy unsafe-best-match --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --upgrade --force-reinstall --no-cache-dir llama-cpp-python==0.3.20",
          "python -c \"import llama_cpp; print(llama_cpp.__version__)\""
        ]
      }
    }
  ]
}
