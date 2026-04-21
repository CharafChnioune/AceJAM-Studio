module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          GRADIO_ANALYTICS_ENABLED: "False",
          GRADIO_SERVER_NAME: "127.0.0.1",
          PYTHONUNBUFFERED: "1",
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
        url: "{{input.event[1]}}"
      }
    }
  ]
}
