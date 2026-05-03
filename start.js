module.exports = {
  daemon: true,
  run: [
    {
      method: "local.set",
      params: {
        port: "{{port}}"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          GRADIO_ANALYTICS_ENABLED: "False",
          GRADIO_SERVER_NAME: "127.0.0.1",
          GRADIO_SERVER_PORT: "{{local.port}}",
          PYTHONUNBUFFERED: "1",
          ACESTEP_LM_BACKEND: "{{platform === 'darwin' && arch === 'arm64' ? 'mlx' : 'pt'}}",
          XDG_CACHE_HOME: "{{path.resolve(cwd, 'cache')}}",
          HF_HOME: "{{path.resolve(cwd, 'cache', 'huggingface')}}",
          HF_MODULES_CACHE: "{{path.resolve(cwd, 'cache', 'hf_modules')}}",
          MPLCONFIGDIR: "{{path.resolve(cwd, 'cache', 'matplotlib')}}"
        },
        path: "app",
        message: [
          "python app.py"
        ],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}",
        bind_url: "{{input.event[1]}}"
      }
    }
  ]
}
