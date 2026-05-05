module.exports = {
  run: [{
    when: "{{exists('app/env')}}",
    method: "fs.rm",
    params: {
      path: "app/env"
    }
  }, {
    when: "{{exists('app/mflux-env')}}",
    method: "fs.rm",
    params: {
      path: "app/mflux-env"
    }
  }, {
    when: "{{exists('app/video-env')}}",
    method: "fs.rm",
    params: {
      path: "app/video-env"
    }
  }, {
    when: "{{exists('app/vendor/mlx-video')}}",
    method: "fs.rm",
    params: {
      path: "app/vendor/mlx-video"
    }
  }, {
    when: "{{exists('app/model_cache')}}",
    method: "fs.rm",
    params: {
      path: "app/model_cache"
    }
  }, {
    when: "{{exists('app/composer_models')}}",
    method: "fs.rm",
    params: {
      path: "app/composer_models"
    }
  }, {
    when: "{{exists('app/data')}}",
    method: "fs.rm",
    params: {
      path: "app/data"
    }
  }, {
    when: "{{exists('cache')}}",
    method: "fs.rm",
    params: {
      path: "cache"
    }
  }, {
    when: "{{exists('app/web/node_modules')}}",
    method: "fs.rm",
    params: {
      path: "app/web/node_modules"
    }
  }, {
    when: "{{exists('app/web/dist')}}",
    method: "fs.rm",
    params: {
      path: "app/web/dist"
    }
  }]
}
